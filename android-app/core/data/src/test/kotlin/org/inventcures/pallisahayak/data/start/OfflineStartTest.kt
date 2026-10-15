package org.inventcures.pallisahayak.data.start

import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.api.generated.ReleaseStatusResponse
import org.inventcures.pallisahayak.data.auth.SessionManager
import org.inventcures.pallisahayak.data.auth.StoredSession
import org.inventcures.pallisahayak.data.auth.TokenStore
import org.inventcures.pallisahayak.data.release.BlockReason
import org.inventcures.pallisahayak.data.release.InMemoryReleaseStatusCache
import org.inventcures.pallisahayak.data.release.ReleaseGate
import org.inventcures.pallisahayak.data.repository.FakeApi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A release build started with no signal opens on the last approved release
 * status, so the Cached Answer Bundle can answer. A release the server was seen
 * to call unapproved stays refused, offline too.
 *
 * Before ticket 23 "can't check" counted as unapproved, so every offline start
 * of a release build ended on the refusal screen and offline answers were
 * unreachable.
 */
class OfflineStartTest {

    private val api = FakeApi()
    private val sessions = SessionManager(
        api,
        object : TokenStore {
            override fun load() = StoredSession(
                userId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
                careRole = "asha_worker",
                siteId = "CCHRC",
                language = "bn-IN",
                accessToken = "access",
                refreshToken = "refresh",
            )

            override fun save(session: StoredSession) = Unit

            override fun clear() = Unit
        },
    )
    private val gate = ReleaseGate(developmentBuild = false)

    private fun release(approved: Boolean, activated: Boolean = approved) = ReleaseStatusResponse(
        activated = activated,
        approved = approved,
        drift = null,
        drifting = false,
        release_id = "study-1@abc1234",
        release_name = "Study 1",
        status = if (approved) "approved" else "draft",
        usable_with_participants = approved && activated,
    )

    @Test
    fun `a release build started with no signal opens on the last approved status`() = runTest {
        val start = AppStart(api, gate, sessions)
        api.release = release(approved = true)
        assertTrue(start.decide() is StartScreen.Ready)

        api.offline = true
        val screen = start.decide()

        assertTrue("offline start was $screen", screen is StartScreen.Ready)
        val session = (screen as StartScreen.Ready).session
        assertEquals("study-1@abc1234", session.releaseId)
        assertTrue(session.releaseApproved)
    }

    @Test
    fun `the last approved status survives a restart through its store`() = runTest {
        val cache = InMemoryReleaseStatusCache()
        api.release = release(approved = true)
        AppStart(api, gate, sessions, cache).decide()

        api.offline = true
        val screen = AppStart(api, gate, sessions, cache).decide()

        assertTrue(screen is StartScreen.Ready)
    }

    @Test
    fun `a release seen unapproved is refused offline too`() = runTest {
        val cache = InMemoryReleaseStatusCache()
        val start = AppStart(api, gate, sessions, cache)
        api.release = release(approved = true)
        start.decide()
        api.release = release(approved = false)
        assertTrue(start.decide() is StartScreen.Refused)

        api.offline = true
        val screen = start.decide()

        assertTrue("offline start was $screen", screen is StartScreen.Refused)
        assertEquals(BlockReason.NOT_APPROVED, (screen as StartScreen.Refused).decision.reason)
    }

    @Test
    fun `an approved but not activated release is not remembered as usable`() = runTest {
        val start = AppStart(api, gate, sessions)
        api.release = release(approved = true, activated = false)
        assertTrue(start.decide() is StartScreen.Refused)

        api.offline = true

        assertTrue(start.decide() is StartScreen.Refused)
    }

    @Test
    fun `with no status ever seen, no signal is still refused`() = runTest {
        api.offline = true

        val screen = AppStart(api, gate, sessions).decide()

        assertTrue(screen is StartScreen.Refused)
    }
}
