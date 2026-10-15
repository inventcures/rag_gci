package org.inventcures.pallisahayak.data.auth

/**
 * What the phone keeps about the person it is registered to.
 *
 * The server hands back a UUID at Registration, never a name or a phone number,
 * so nothing here identifies anyone off the device. The tokens are null once a
 * session has ended (a refresh was refused): the device still knows who it
 * belongs to, so the PIN alone signs them back in.
 */
data class StoredSession(
    val userId: String,
    /** The Care Role's wire name, as registered. */
    val careRole: String,
    val siteId: String,
    /** BCP-47 tag of the language registered with, e.g. `hi-IN`. */
    val language: String,
    val accessToken: String? = null,
    val refreshToken: String? = null,
    /** When the access token expires, in Unix seconds, as the server reported it. */
    val accessExpiresAt: Double? = null,
) {
    /** True when there is a token pair to call the API with. */
    val signedIn: Boolean get() = accessToken != null && refreshToken != null

    /** The same registration with its tokens dropped. */
    fun signedOut(): StoredSession = copy(accessToken = null, refreshToken = null, accessExpiresAt = null)
}

/**
 * Where the session is kept between launches.
 *
 * On a phone this is [KeystoreTokenStore], which encrypts with a key held in the
 * Android Keystore. Kept behind an interface so the session logic can be tested
 * without a Keystore.
 */
interface TokenStore {
    /** The stored session, or null on a phone that has never registered. */
    fun load(): StoredSession?

    fun save(session: StoredSession)

    fun clear()
}

/** Where the participant is in signing in. */
sealed interface AuthState {
    /** Nobody has registered on this phone. */
    data object Unregistered : AuthState

    /** Registered, but the session ended: the PIN is needed again. */
    data class NeedsPin(val userId: String) : AuthState

    /** Signed in; calls carry this session's access token. */
    data class SignedIn(val session: StoredSession) : AuthState
}

/** The outcome of registering or signing in. */
sealed interface AuthResult {
    /** The phone now holds a token pair. */
    data object SignedIn : AuthResult

    /**
     * The server said no: a wrong PIN, a locked account or a refused registration.
     *
     * A locked account gets the same answer as a wrong PIN (report 08), so the
     * app cannot tell them apart and does not pretend to.
     */
    data object Refused : AuthResult

    /** The server could not be reached. Nothing changed on the phone. */
    data object Unreachable : AuthResult
}
