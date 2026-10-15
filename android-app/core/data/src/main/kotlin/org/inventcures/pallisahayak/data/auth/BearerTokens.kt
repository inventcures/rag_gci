package org.inventcures.pallisahayak.data.auth

import okhttp3.Authenticator
import okhttp3.Interceptor
import okhttp3.Request
import okhttp3.Response
import okhttp3.Route

/** The header every authenticated mobile API call carries. */
internal const val AUTHORIZATION = "Authorization"

internal fun bearer(token: String): String = "Bearer $token"

/**
 * Signs every call with the participant's access token.
 *
 * Also signs the voice socket's upgrade request, since OkHttp runs application
 * interceptors for a WebSocket handshake too. A call made before sign-in goes
 * out unsigned and the server answers 401, which is the honest result.
 */
class BearerTokenInterceptor(private val sessions: SessionManager) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val token = sessions.accessToken()
        if (token == null || request.header(AUTHORIZATION) != null) {
            return chain.proceed(request)
        }
        return chain.proceed(request.newBuilder().header(AUTHORIZATION, bearer(token)).build())
    }
}

/**
 * Refreshes an expired access token once, and retries the call with the new one.
 *
 * OkHttp calls this on a 401. The retry is attempted only for the first 401 of
 * a call: if the retried call is refused too, or the refresh is refused, the
 * call fails with that 401 and the session manager has already sent the
 * participant back to the PIN.
 */
class TokenRefreshAuthenticator(private val sessions: SessionManager) : Authenticator {
    override fun authenticate(route: Route?, response: Response): Request? {
        if (response.priorResponse != null) return null
        val rejected = response.request.header(AUTHORIZATION)
            ?.removePrefix("Bearer ")
            ?: return null
        val fresh = sessions.refreshAfterRejection(rejected) ?: return null
        return response.request.newBuilder().header(AUTHORIZATION, bearer(fresh)).build()
    }
}
