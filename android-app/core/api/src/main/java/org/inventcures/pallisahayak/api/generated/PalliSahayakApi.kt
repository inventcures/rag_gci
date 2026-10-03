package org.inventcures.pallisahayak.api.generated

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query
import retrofit2.http.Multipart
import retrofit2.http.Part
import okhttp3.RequestBody

/**
 * Generated from the API contract. Do not edit by hand.
 *
 * A renamed or removed server field is caught by verifyApiContract rather
 * than at runtime, which is the point: the failure that matters here is
 * a client that compiles against a schema nobody serves, returning empty
 * answers with no visible error.
 */
interface PalliSahayakApi {

    @POST(
        "/api/mobile/v1/query"
    )
    suspend fun query(@Body body: MobileQueryRequest): Response<MobileQueryResponse>

    @Multipart
    @POST(
        "/api/mobile/v1/query/voice"
    )
    suspend fun voiceQuery(@Query("language") language: String, @Part("audio") audio: RequestBody): Response<VoiceQueryResponse>

    @GET(
        "/api/mobile/v1/cache/bundle"
    )
    suspend fun cacheBundle(@Query("language") language: String): Response<CacheBundleResponse>

    @POST(
        "/api/mobile/v1/auth/login"
    )
    suspend fun login(@Body body: LoginRequest): Response<AuthResponse>
}
