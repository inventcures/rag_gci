package org.inventcures.pallisahayak.data.sync.wire

import com.squareup.moshi.Json
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

/**
 * The server's sync push (`POST /api/mobile/v1/sync/push`, `mobile_api/router.py`).
 *
 * Written by hand because sync is not a generated operation: the push takes free
 * dictionaries per kind. The shape is the one report 09 fixes, and
 * `SyncQueueTest` drives it through a fake of this interface. It goes through
 * the signed client, so every push carries the participant's access token.
 */
interface SyncApi {
    @POST("/api/mobile/v1/sync/push")
    suspend fun push(@Body body: SyncPushBody): Response<SyncPushResult>
}

/** One push. Each list is in queue order; an item's place in its list is its index. */
data class SyncPushBody(
    @Json(name = "observations") val observations: List<Map<String, Any?>> = emptyList(),
    @Json(name = "medication_reminders") val medication_reminders: List<Map<String, Any?>> = emptyList(),
    @Json(name = "interaction_logs") val interaction_logs: List<Map<String, Any?>> = emptyList(),
    @Json(name = "vignette_responses") val vignette_responses: List<Map<String, Any?>> = emptyList(),
    @Json(name = "device_timestamp") val device_timestamp: Double,
)

/** The server's answer to a push: one acknowledgement per item pushed. */
data class SyncPushResult(
    @Json(name = "accepted") val accepted: Int,
    @Json(name = "rejected") val rejected: Int,
    @Json(name = "acknowledgements") val acknowledgements: List<SyncAcknowledgement>,
    @Json(name = "conflicts") val conflicts: List<Map<String, Any?>> = emptyList(),
)

/**
 * The server's answer for one item, matched to it by ([kind], [index]).
 *
 * [reason] is set only when [status] is "rejected": "invalid" and "not_owned"
 * will never be accepted; "save_failed", "unavailable" and "over_limit" may be
 * on a later push.
 */
data class SyncAcknowledgement(
    @Json(name = "kind") val kind: String,
    @Json(name = "index") val index: Int,
    @Json(name = "id") val id: String? = null,
    @Json(name = "status") val status: String,
    @Json(name = "reason") val reason: String? = null,
)
