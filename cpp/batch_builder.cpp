/**
 * High-Performance C++ Batch Builder for Tiny-Seq2Seq.
 * 
 * Implements:
 * - Direct memory-buffer access over flat uint16 token arrays and int64 offsets
 * - Fast length bucketing and token-budget bin packing
 * - Source reversal and special token packing (<bos>, <eos>, <pad>)
 * - GIL release (py::gil_scoped_release) for concurrent background prefetching
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

class CppBatchBuilder {
public:
    CppBatchBuilder(
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> src_data,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> src_offsets,
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> tgt_data,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> tgt_offsets
    ) {
        // Keep NumPy buffer views
        src_data_ = src_data;
        src_offsets_ = src_offsets;
        tgt_data_ = tgt_data;
        tgt_offsets_ = tgt_offsets;

        auto src_off_info = src_offsets_.request();
        auto tgt_off_info = tgt_offsets_.request();

        if (src_off_info.size != tgt_off_info.size) {
            throw std::runtime_error("Source and target offset arrays have different lengths.");
        }

        num_sentences_ = src_off_info.size > 0 ? (src_off_info.size - 1) : 0;

        src_tokens_ptr_ = static_cast<const uint16_t*>(src_data_.request().ptr);
        src_offsets_ptr_ = static_cast<const int64_t*>(src_off_info.ptr);
        tgt_tokens_ptr_ = static_cast<const uint16_t*>(tgt_data_.request().ptr);
        tgt_offsets_ptr_ = static_cast<const int64_t*>(tgt_off_info.ptr);
    }

    size_t get_num_sentences() const {
        return num_sentences_;
    }

    /**
     * Plans batches according to token budget and sequence length bucketing.
     * Releases GIL during sorting and packing.
     */
    std::vector<std::vector<int64_t>> plan_batches(
        int64_t token_budget,
        bool shuffle,
        uint64_t seed
    ) {
        std::vector<std::vector<int64_t>> batches;

        {
            py::gil_scoped_release release;

            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(0.0, 0.999);

            struct Item {
                int64_t index;
                int64_t max_len;
                double sort_key;
            };

            std::vector<Item> items(num_sentences_);
            for (size_t i = 0; i < num_sentences_; ++i) {
                int64_t s_len = (src_offsets_ptr_[i + 1] - src_offsets_ptr_[i]) + 1; // +1 for EOS
                int64_t t_len = (tgt_offsets_ptr_[i + 1] - tgt_offsets_ptr_[i]) + 1; // +1 for EOS/BOS
                int64_t m_len = std::max(s_len, t_len);
                double key = shuffle ? (static_cast<double>(m_len) + dist(rng)) : static_cast<double>(m_len);
                items[i] = {static_cast<int64_t>(i), m_len, key};
            }

            // Sort by length (with random tie-breaking if shuffle is on)
            std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
                return a.sort_key < b.sort_key;
            });

            std::vector<int64_t> current_batch;
            int64_t current_max_src = 0;
            int64_t current_max_tgt = 0;

            for (const auto& item : items) {
                int64_t idx = item.index;
                int64_t s_len = (src_offsets_ptr_[idx + 1] - src_offsets_ptr_[idx]) + 1;
                int64_t t_len = (tgt_offsets_ptr_[idx + 1] - tgt_offsets_ptr_[idx]) + 1;

                int64_t new_max_src = std::max(current_max_src, s_len);
                int64_t new_max_tgt = std::max(current_max_tgt, t_len);
                int64_t new_batch_size = static_cast<int64_t>(current_batch.size()) + 1;
                int64_t new_tokens = std::max(new_max_src, new_max_tgt) * new_batch_size;

                if (!current_batch.empty() && new_tokens > token_budget) {
                    batches.push_back(current_batch);
                    current_batch.clear();
                    current_batch.push_back(idx);
                    current_max_src = s_len;
                    current_max_tgt = t_len;
                } else {
                    current_batch.push_back(idx);
                    current_max_src = new_max_src;
                    current_max_tgt = new_max_tgt;
                }
            }

            if (!current_batch.empty()) {
                batches.push_back(current_batch);
            }

            if (shuffle) {
                std::shuffle(batches.begin(), batches.end(), rng);
            }
        }

        return batches;
    }

    /**
     * Builds and collates a batch into pre-allocated NumPy contiguous 2D arrays.
     * Implements source reversal and BOS/EOS insertion.
     */
    py::dict build_batch(
        const std::vector<int64_t>& indices,
        bool reverse_source,
        int64_t pad_id,
        int64_t bos_id,
        int64_t eos_id
    ) {
        size_t batch_size = indices.size();
        if (batch_size == 0) {
            throw std::runtime_error("Cannot build an empty batch.");
        }

        int64_t max_src_len = 0;
        int64_t max_tgt_len = 0;

        for (int64_t idx : indices) {
            int64_t s_len = (src_offsets_ptr_[idx + 1] - src_offsets_ptr_[idx]) + 1; // +1 for EOS
            int64_t t_len = (tgt_offsets_ptr_[idx + 1] - tgt_offsets_ptr_[idx]) + 1; // +1 for EOS/BOS
            if (s_len > max_src_len) max_src_len = s_len;
            if (t_len > max_tgt_len) max_tgt_len = t_len;
        }

        py::ssize_t b_sz = static_cast<py::ssize_t>(batch_size);
        py::ssize_t s_len_sz = static_cast<py::ssize_t>(max_src_len);
        py::ssize_t t_len_sz = static_cast<py::ssize_t>(max_tgt_len);

        // Allocate NumPy return arrays
        py::array_t<int64_t> src_ids({b_sz, s_len_sz});
        py::array_t<int64_t> src_lens({b_sz});
        py::array_t<int64_t> tgt_in_ids({b_sz, t_len_sz});
        py::array_t<int64_t> tgt_out_ids({b_sz, t_len_sz});
        py::array_t<int64_t> tgt_lens({b_sz});


        auto src_ids_ptr = static_cast<int64_t*>(src_ids.request().ptr);
        auto src_lens_ptr = static_cast<int64_t*>(src_lens.request().ptr);
        auto tgt_in_ids_ptr = static_cast<int64_t*>(tgt_in_ids.request().ptr);
        auto tgt_out_ids_ptr = static_cast<int64_t*>(tgt_out_ids.request().ptr);
        auto tgt_lens_ptr = static_cast<int64_t*>(tgt_lens.request().ptr);

        int64_t total_tokens = 0;

        {
            py::gil_scoped_release release;

            // Initialize all token matrices with pad_id
            std::fill_n(src_ids_ptr, batch_size * max_src_len, pad_id);
            std::fill_n(tgt_in_ids_ptr, batch_size * max_tgt_len, pad_id);
            std::fill_n(tgt_out_ids_ptr, batch_size * max_tgt_len, pad_id);

            for (size_t i = 0; i < batch_size; ++i) {
                int64_t idx = indices[i];

                int64_t s_start = src_offsets_ptr_[idx];
                int64_t s_end = src_offsets_ptr_[idx + 1];
                int64_t s_len = s_end - s_start;

                int64_t t_start = tgt_offsets_ptr_[idx];
                int64_t t_end = tgt_offsets_ptr_[idx + 1];
                int64_t t_len = t_end - t_start;

                int64_t* row_src = src_ids_ptr + (i * max_src_len);
                int64_t* row_tgt_in = tgt_in_ids_ptr + (i * max_tgt_len);
                int64_t* row_tgt_out = tgt_out_ids_ptr + (i * max_tgt_len);

                // Source Reversal:
                if (reverse_source) {
                    for (int64_t j = 0; j < s_len; ++j) {
                        row_src[j] = static_cast<int64_t>(src_tokens_ptr_[s_end - 1 - j]);
                    }
                } else {
                    for (int64_t j = 0; j < s_len; ++j) {
                        row_src[j] = static_cast<int64_t>(src_tokens_ptr_[s_start + j]);
                    }
                }
                row_src[s_len] = eos_id;
                src_lens_ptr[i] = s_len + 1;

                // Target Input: BOS + target
                row_tgt_in[0] = bos_id;
                for (int64_t j = 0; j < t_len; ++j) {
                    row_tgt_in[1 + j] = static_cast<int64_t>(tgt_tokens_ptr_[t_start + j]);
                }

                // Target Output: target + EOS
                for (int64_t j = 0; j < t_len; ++j) {
                    row_tgt_out[j] = static_cast<int64_t>(tgt_tokens_ptr_[t_start + j]);
                }
                row_tgt_out[t_len] = eos_id;
                tgt_lens_ptr[i] = t_len + 1;

                total_tokens += (t_len + 1);
            }
        }

        py::dict result;
        result["src_ids"] = src_ids;
        result["src_lens"] = src_lens;
        result["tgt_in_ids"] = tgt_in_ids;
        result["tgt_out_ids"] = tgt_out_ids;
        result["tgt_lens"] = tgt_lens;
        result["num_tokens"] = total_tokens;

        return result;
    }

private:
    py::array_t<uint16_t> src_data_;
    py::array_t<int64_t> src_offsets_;
    py::array_t<uint16_t> tgt_data_;
    py::array_t<int64_t> tgt_offsets_;

    const uint16_t* src_tokens_ptr_;
    const int64_t* src_offsets_ptr_;
    const uint16_t* tgt_tokens_ptr_;
    const int64_t* tgt_offsets_ptr_;
    size_t num_sentences_;
};

PYBIND11_MODULE(seq2seq_c_batcher, m) {
    m.doc() = "High-performance C++ length-bucketed batch builder for Tiny-Seq2Seq";

    py::class_<CppBatchBuilder>(m, "CppBatchBuilder")
        .def(py::init<
            py::array_t<uint16_t, py::array::c_style | py::array::forcecast>,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
            py::array_t<uint16_t, py::array::c_style | py::array::forcecast>,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        >())
        .def("get_num_sentences", &CppBatchBuilder::get_num_sentences)
        .def("plan_batches", &CppBatchBuilder::plan_batches,
             py::arg("token_budget") = 4000,
             py::arg("shuffle") = true,
             py::arg("seed") = 42)
        .def("build_batch", &CppBatchBuilder::build_batch,
             py::arg("indices"),
             py::arg("reverse_source") = true,
             py::arg("pad_id") = 0,
             py::arg("bos_id") = 2,
             py::arg("eos_id") = 3);
}
