package org.inventcures.pallisahayak.data.auth

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.runBlocking
import org.inventcures.pallisahayak.api.generated.AuthResponse
import org.inventcures.pallisahayak.api.generated.DeviceRegistrationRequest
import org.inventcures.pallisahayak.api.generated.LoginRequest
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.api.generated.RefreshRequest
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import retrofit2.Response
import java.io.IOException

/**
 * Registration, PIN sign-in and token refresh against the mobile API.
 *
 * The server's contract (review 3, report 08): register and login return an
 * access token and a refresh token. Every other call carries the access token
 * as a bearer token. `/auth/refresh` takes the refresh token in the body and no
 * bearer header, and refuses with 401 once the session has ended, which is at
 * most six days after the PIN was entered. A refused refresh sends the
 * participant back to the PIN.
 *
 * [api] must be a client without [BearerTokenInterceptor] and
 * [TokenRefreshAuthenticator]: these calls are what produce the tokens, and a
 * refresh that went through the authenticator could recurse.
 */
class SessionManager(
    private val api: PalliSahayakApi,
    private val store: TokenStore,
) {
    private val _state = MutableStateFlow(stateOf(store.load()))

    /** Where the participant is in signing in. The app shows a screen per state. */
    val state: StateFlow<AuthState> = _state.asStateFlow()

    /** The access token to send, or null when not signed in. */
    fun accessToken(): String? = (state.value as? AuthState.SignedIn)?.session?.accessToken

    /**
     * Register this phone to a participant with their Care Role and a PIN.
     *
     * No name and no phone number are sent. The server requires both fields and
     * reads neither; the phone has no reason to hold either, and Identifiable
     * data that is never collected cannot leak.
     */
    suspend fun register(
        role: CareRole,
        pin: String,
        language: Language,
        siteId: String,
    ): AuthResult {
        val response = call {
            api.register(
                DeviceRegistrationRequest(
                    name = "",
                    phone_hash = "",
                    role = role.wireName,
                    language = language.tag,
                    site_id = siteId,
                    pin = pin,
                ),
            )
        } ?: return AuthResult.Unreachable
        val pair = response.body()
        if (!response.isSuccessful || pair == null) return AuthResult.Refused
        signIn(
            StoredSession(
                userId = pair.user_id,
                careRole = role.wireName,
                siteId = siteId,
                language = language.tag,
            ),
            pair,
        )
        return AuthResult.SignedIn
    }

    /** Sign the registered participant back in with their PIN. */
    suspend fun signIn(pin: String): AuthResult {
        val registered = store.load() ?: return AuthResult.Refused
        val response = call { api.login(LoginRequest(user_id = registered.userId, pin = pin)) }
            ?: return AuthResult.Unreachable
        val pair = response.body()
        if (!response.isSuccessful || pair == null) return AuthResult.Refused
        signIn(registered, pair)
        return AuthResult.SignedIn
    }

    /**
     * Renew the session after the server refused [rejected], at most once per rejection.
     *
     * Called by [TokenRefreshAuthenticator] on OkHttp's thread, so it blocks.
     * Synchronised so that several calls refused at once trigger one refresh: a
     * caller whose token has already been replaced gets the new one.
     *
     * Returns the access token to retry with, or null. A refused refresh ends the
     * session and the state becomes [AuthState.NeedsPin]. A refresh that could
     * not reach the server changes nothing, since being offline is not a reason
     * to ask for the PIN.
     */
    @Synchronized
    fun refreshAfterRejection(rejected: String): String? {
        val current = (state.value as? AuthState.SignedIn)?.session ?: return null
        if (current.accessToken != rejected) return current.accessToken
        val refreshToken = current.refreshToken ?: return null

        val response = runBlocking { call { api.refresh(RefreshRequest(refresh_token = refreshToken)) } }
            ?: return null
        val pair = response.body()
        if (!response.isSuccessful || pair == null) {
            endSession(current)
            return null
        }
        return signIn(current, pair).accessToken
    }

    private fun signIn(registration: StoredSession, pair: AuthResponse): StoredSession {
        val session = registration.copy(
            userId = pair.user_id,
            accessToken = pair.token,
            refreshToken = pair.refresh_token,
            accessExpiresAt = pair.expires_at,
        )
        store.save(session)
        _state.value = AuthState.SignedIn(session)
        return session
    }

    private fun endSession(session: StoredSession) {
        val signedOut = session.signedOut()
        store.save(signedOut)
        _state.value = AuthState.NeedsPin(signedOut.userId)
    }

    /** Run a call, or return null when the server could not be reached. */
    private suspend fun <T> call(block: suspend () -> Response<T>): Response<T>? =
        try {
            block()
        } catch (error: IOException) {
            null
        }

    private companion object {
        fun stateOf(stored: StoredSession?): AuthState = when {
            stored == null -> AuthState.Unregistered
            stored.signedIn -> AuthState.SignedIn(stored)
            else -> AuthState.NeedsPin(stored.userId)
        }
    }
}
