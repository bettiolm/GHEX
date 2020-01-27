/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_GLUE_ATLAS_GPU_STORAGE_HPP
#define INCLUDED_GHEX_GLUE_ATLAS_GPU_STORAGE_HPP

#include <vector>
#include <memory>

#include <atlas/field.h>

#include "../../allocator/unified_memory_allocator.hpp"


namespace gridtools {

    namespace ghex {

        template<typename T>
        class device_copy {

            private:

                using allocator_type = gridtools::ghex::allocator::unified_memory_allocator<T>;

                struct device_copy_impl {

                        std::vector<T, allocator_type> m_data;
                        T* m_data_ptr;
                        atlas::idx_t m_levels;

                        device_copy_impl(const atlas::Field& field) :
                            m_data{field.data<T>(), field.data<T>() + field.size()},
                            m_data_ptr{&(m_data[0])},
                            m_levels{field.levels()} {}

                        device_copy_impl(const device_copy_impl& other) = default;
                        device_copy_impl(device_copy_impl&& other) = default;

                        device_copy_impl& operator = (const device_copy_impl& other) = default;
                        device_copy_impl& operator = (device_copy_impl&& other) = default;

                        std::size_t size() const { return m_data.size(); }
                        atlas::idx_t nodes() const { return m_data.size() / m_levels; }
                        atlas::idx_t levels() const { return m_levels; }

                        ATLAS_HOST_DEVICE T& get(atlas::idx_t node, atlas::idx_t level) {
                            return m_data_ptr[node * m_levels + level];
                        }
                        ATLAS_HOST_DEVICE const T& get(atlas::idx_t node, atlas::idx_t level) const {
                            return m_data_ptr[node * m_levels + level];
                        }

                };

                device_copy_impl* m_device_copy_impl;

            public:

                device_copy(const atlas::Field& field) :
                    m_device_copy_impl{ new(allocator_type{}.allocate(sizeof(device_copy_impl))) device_copy_impl{field} } {}

                device_copy(const device_copy& other) = delete;
                device_copy(device_copy&& other) = default;

                device_copy& operator = (const device_copy& other) = delete;
                device_copy& operator = (device_copy&& other) = default;

                ~device_copy() {
                    delete m_device_copy_impl;
                    m_device_copy_impl = nullptr;
                }

                std::size_t size() const { return m_device_copy_impl->size(); }
                atlas::idx_t nodes() const { return m_device_copy_impl->nodes(); }
                atlas::idx_t levels() const { return m_device_copy_impl->levels(); }

                ATLAS_HOST_DEVICE T& operator()(atlas::idx_t node, atlas::idx_t level) {
                    return m_device_copy_impl->get(node, level);
                }
                ATLAS_HOST_DEVICE const T& operator()(atlas::idx_t node, atlas::idx_t level) const {
                    return m_device_copy_impl->get(node, level);
                }

        }; // class device_copy

        template<typename T>
        device_copy<T>* make_device_copy(const atlas::Field& field) {
            using allocator_type = gridtools::ghex::allocator::unified_memory_allocator<T>;
            return new(allocator_type{}.allocate(sizeof(device_copy<T>))) device_copy<T>{field};
        }

        template<typename T>
        void update_host_field(const device_copy<T>& d_copy, atlas::Field field) {
            auto h_view = atlas::array::make_host_view<T, 2>(field);
            for(auto node = 0; node < d_copy.nodes(); ++node) {
                for(auto level = 0; level < d_copy.levels(); ++level) {
                    h_view(node, level) = d_copy(node, level);
                }
            }
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_GLUE_ATLAS_GPU_STORAGE_HPP */
