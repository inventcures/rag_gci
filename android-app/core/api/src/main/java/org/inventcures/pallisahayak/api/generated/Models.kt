package org.inventcures.pallisahayak.api.generated

import com.squareup.moshi.Json

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class AuthResponse(
    @Json(name = "expires_at") val expires_at: Double,
    @Json(name = "refresh_token") val refresh_token: String,
    @Json(name = "token") val token: String,
    @Json(name = "user_id") val user_id: String,
)

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class CacheBundleResponse(
    @Json(name = "emergency_keywords") val emergency_keywords: Map<String, Any?>,
    @Json(name = "evidence_badge_metadata") val evidence_badge_metadata: Map<String, Any?>,
    @Json(name = "generated_at") val generated_at: Double,
    @Json(name = "language") val language: String,
    @Json(name = "queries") val queries: List<Map<String, Any?>>,
    @Json(name = "treatments") val treatments: List<Map<String, Any?>>,
    @Json(name = "version") val version: String,
)

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class LoginRequest(
    @Json(name = "pin") val pin: String,
    @Json(name = "user_id") val user_id: String,
)

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class MobileQueryRequest(
    @Json(name = "include_context") val include_context: Boolean? = null,
    @Json(name = "language") val language: String? = null,
    @Json(name = "patient_id") val patient_id: Any?? = null,
    @Json(name = "query") val query: String,
)

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class MobileQueryResponse(
    @Json(name = "answer") val answer: String,
    @Json(name = "confidence") val confidence: Double,
    @Json(name = "disclaimer") val disclaimer: Any?? = null,
    @Json(name = "emergency_level") val emergency_level: String,
    @Json(name = "evidence_level") val evidence_level: String,
    @Json(name = "sources") val sources: List<Map<String, Any?>>,
    @Json(name = "validation_status") val validation_status: String,
)

/**
 * Generated from the API contract. Do not edit by hand; run
 * scripts/generate_kotlin_client.py instead.
 */
data class VoiceQueryResponse(
    @Json(name = "answer") val answer: String,
    @Json(name = "audio_base64") val audio_base64: Any?? = null,
    @Json(name = "confidence") val confidence: Double,
    @Json(name = "disclaimer") val disclaimer: Any?? = null,
    @Json(name = "emergency_level") val emergency_level: String,
    @Json(name = "evidence_level") val evidence_level: String,
    @Json(name = "sources") val sources: List<Map<String, Any?>>,
    @Json(name = "transcript") val transcript: String,
    @Json(name = "validation_status") val validation_status: String,
)
