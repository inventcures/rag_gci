package org.inventcures.pallisahayak.data.start

import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.api.generated.ReleaseStatusResponse
import org.inventcures.pallisahayak.data.auth.AuthState
import org.inventcures.pallisahayak.data.auth.SessionManager
import org.inventcures.pallisahayak.data.release.GateDecision
import org.inventcures.pallisahayak.data.release.InMemoryReleaseStatusCache
import org.inventcures.pallisahayak.data.release.ReleaseGate
import org.inventcures.pallisahayak.data.release.ReleaseStatus
import org.inventcures.pallisahayak.data.release.ReleaseStatusCache
import org.inventcures.pallisahayak.data.repository.SessionContext
import java.io.IOException

/** The first screen the app shows, and what it needs to show it. */
sealed interface StartScreen {
    /** The release is not approved for participants. Nothing else is offered. */
    data class Refused(val decision: GateDecision.Blocked) : StartScreen

    /** Nobody has registered on this phone: Care Role and PIN. */
    data object Register : StartScreen

    /** Registered, but the session ended: the PIN. */
    data object SignIn : StartScreen

    /** Signed in. Every interaction is recorded against [session]. */
    data class Ready(val session: SessionContext) : StartScreen
}

/**
 * What the app does when it starts.
 *
 * The release gate runs first, before registration is offered (criterion 6):
 * a build the server does not confirm as approved and activated opens on the
 * refusal screen. When the server cannot be reached, the last status it
 * confirmed as usable for this build stands in ([ReleaseStatusCache]), so a
 * phone with no signal can still answer from the Cached Answer Bundle; with no
 * such status, no signal counts as not approved. Then the sign-in state decides
 * between registration, the PIN and the app.
 *
 * [api] is the unsigned client: the release status is read before there is a
 * session (`auth/policy.py`).
 */
class AppStart(
    private val api: PalliSahayakApi,
    private val gate: ReleaseGate,
    private val sessions: SessionManager,
    private val releaseCache: ReleaseStatusCache = InMemoryReleaseStatusCache(),
) {
    /** Ask the server about this release, and decide the first screen. */
    suspend fun decide(): StartScreen = screenFor(release(), sessions.state.value)

    /**
     * The server's view of the running release. When the server cannot say, the
     * last usable status it gave, else [ReleaseStatus.unknown].
     *
     * Only a status the server actually returned updates the cache: a usable one
     * is remembered, and any other clears it, so a release seen unapproved (or
     * drifting, or deactivated) is refused offline as well.
     */
    suspend fun release(): ReleaseStatus {
        val fetched = fetchRelease() ?: return releaseCache.lastApproved() ?: ReleaseStatus.unknown()
        if (fetched.usableWithParticipants && !fetched.drifting) {
            releaseCache.remember(fetched)
        } else {
            releaseCache.forget()
        }
        return fetched
    }

    /** The status the server returned, or null when it could not be asked. */
    private suspend fun fetchRelease(): ReleaseStatus? {
        val response = try {
            api.releaseStatus()
        } catch (error: IOException) {
            return null
        }
        val body = response.body()
        return if (response.isSuccessful && body != null) body.toReleaseStatus() else null
    }

    /** The screen for a release status and a sign-in state. */
    fun screenFor(release: ReleaseStatus, auth: AuthState): StartScreen {
        val decision = gate.evaluate(release)
        if (decision is GateDecision.Blocked) return StartScreen.Refused(decision)
        return when (auth) {
            AuthState.Unregistered -> StartScreen.Register
            is AuthState.NeedsPin -> StartScreen.SignIn
            is AuthState.SignedIn -> StartScreen.Ready(
                SessionContext(
                    participantId = auth.session.userId,
                    careRole = auth.session.careRole,
                    siteId = auth.session.siteId,
                    language = auth.session.language,
                    releaseId = release.releaseId,
                    releaseApproved = release.usableWithParticipants,
                ),
            )
        }
    }
}

/** The generated wire model as the gate reads it. */
internal fun ReleaseStatusResponse.toReleaseStatus(): ReleaseStatus = ReleaseStatus(
    releaseId = release_id,
    status = status,
    activated = activated,
    approved = approved,
    usableWithParticipants = usable_with_participants,
    drifting = drifting,
    drift = drift.orEmpty().map { finding -> finding["component"]?.toString().orEmpty() },
)
