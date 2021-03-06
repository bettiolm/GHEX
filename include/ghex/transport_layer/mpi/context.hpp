/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_TL_MPI_CONTEXT_HPP
#define INCLUDED_TL_MPI_CONTEXT_HPP

#include <mutex>
#include "../context.hpp"
#include "./communicator.hpp"
#include "../communicator.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace mpi {

            struct transport_context
            {
                using tag = mpi_tag;
                using communicator_type = tl::communicator<mpi::communicator>;
                using shared_state_type = typename communicator_type::shared_state_type;
                using state_type = typename communicator_type::state_type;
                using state_ptr = std::unique_ptr<state_type>;
                using state_vector = std::vector<state_ptr>;

                const mpi::rank_topology& m_rank_topology;
                MPI_Comm m_comm;
                shared_state_type m_shared_state;
                state_type m_state;
                state_vector m_states;
                std::mutex m_mutex;

                transport_context(const rank_topology& t)
                    : m_rank_topology{t}
                    , m_comm{t.mpi_comm()}
                    , m_shared_state(m_rank_topology)
                {}

                MPI_Comm mpi_comm() const { return m_comm; }

                communicator_type get_serial_communicator()
                {
                    return {&m_shared_state, &m_state};
                }

                communicator_type get_communicator()
                {
                    std::lock_guard<std::mutex> lock(m_mutex); // we need to guard only the insertion in the vector,
                                                               // but this is not a performance critical section
                    m_states.push_back(std::make_unique<state_type>());
                    return {&m_shared_state, m_states[m_states.size()-1].get()};
                }
            };

            } // namespace mpi

            template<>
            struct context_factory<mpi_tag>
            {
                using context_type = context<mpi::transport_context>;
                static std::unique_ptr<context_type> create(MPI_Comm mpi_comm)
                {
                    auto new_comm = detail::clone_mpi_comm(mpi_comm);
                    return std::unique_ptr<context_type>{
                        new context_type{new_comm}};
                }
            };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_TL_MPI_CONTEXT_HPP */
