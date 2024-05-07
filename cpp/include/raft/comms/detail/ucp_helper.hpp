/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/util/cudart_utils.hpp>

#include <stdio.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>

namespace raft {
namespace comms {
namespace detail {

/**
 * Standard UCX request object that will be passed
 * around asynchronously. This object is really
 * opaque and the comms layer only cares that it
 * has been completed. Because raft comms do not
 * initialize the ucx application context, it doesn't
 * own this object and thus it's important not to
 * modify this struct.
 */
struct ucx_context {
  int completed;
};

/**
 * Wraps the `ucx_context` request and adds a few
 * other fields for trace logging and cleanup.
 */
class ucp_request {
 public:
  struct ucx_context* req;
  bool needs_release = true;
};

// by default, match the whole tag
static const ucp_tag_t default_tag_mask = (ucp_tag_t)-1;

/**
 * @brief Asynchronous send callback sets request to completed
 */
static void send_callback(void* request, ucs_status_t status)
{
  struct ucx_context* context = (struct ucx_context*)request;
  context->completed          = 1;
}

/**
 * @brief Asynchronous recv callback sets request to completed
 */
static void recv_callback(void* request, ucs_status_t status, ucp_tag_recv_info_t* info)
{
  struct ucx_context* context = (struct ucx_context*)request;
  context->completed          = 1;
}

ucp_tag_t build_message_tag(int rank, int tag)
{
  // keeping the rank in the lower bits enables debugging.
  return ((uint32_t)tag << 31) | (uint32_t)rank;
}

/**
 * Helper class for interacting with ucp.
 */
class comms_ucp_handler {
 public:
  /**
   * @brief Frees any memory underlying the given ucp request object
   */
  void free_ucp_request(ucp_request* request) const
  {
    if (request->needs_release) {
      request->req->completed = 0;
      ucp_request_free(request->req);
    }
    free(request);
  }

  /**
   * @brief Asynchronously send data to the given endpoint using the given tag
   */
  void ucp_isend(ucp_request* req,
                 ucp_ep_h ep_ptr,
                 const void* buf,
                 size_t size,
                 int tag,
                 ucp_tag_t tag_mask,
                 int rank) const
  {
    ucp_tag_t ucp_tag = build_message_tag(rank, tag);

    ucs_status_ptr_t send_result =
      ucp_tag_send_nb(ep_ptr, buf, size, ucp_dt_make_contig(1), ucp_tag, send_callback);
    struct ucx_context* ucp_req = (struct ucx_context*)send_result;

    if (UCS_PTR_IS_ERR(send_result)) {
      ASSERT(!UCS_PTR_IS_ERR(send_result),
             "unable to send UCX data message (%d)\n",
             UCS_PTR_STATUS(send_result));
      /**
       * If the request didn't fail, but it's not OK, it is in flight.
       * Expect the handler to be invoked
       */
    } else if (UCS_PTR_STATUS(send_result) != UCS_OK) {
      /**
       * If the request is OK, it's already been completed and we don't need to wait on it.
       * The request will be a nullptr, however, so we need to create a new request
       * and set it to completed to make the "waitall()" function work properly.
       */
      req->needs_release = true;
    } else {
      req->needs_release = false;
    }

    req->req = ucp_req;
  }

  /**
   * @brief Asynchronously receive data from given endpoint with the given tag.
   */
  void ucp_irecv(ucp_request* req,
                 ucp_worker_h worker,
                 ucp_ep_h ep_ptr,
                 void* buf,
                 size_t size,
                 int tag,
                 ucp_tag_t tag_mask,
                 int sender_rank) const
  {
    ucp_tag_t ucp_tag = build_message_tag(sender_rank, tag);

    ucs_status_ptr_t recv_result =
      ucp_tag_recv_nb(worker, buf, size, ucp_dt_make_contig(1), ucp_tag, tag_mask, recv_callback);

    struct ucx_context* ucp_req = (struct ucx_context*)recv_result;

    req->req           = ucp_req;
    req->needs_release = true;

    ASSERT(!UCS_PTR_IS_ERR(recv_result),
           "unable to receive UCX data message (%d)\n",
           UCS_PTR_STATUS(recv_result));
  }
};
}  // end namespace detail
}  // end namespace comms
}  // end namespace raft
