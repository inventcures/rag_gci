package org.inventcures.pallisahayak.data.release

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The release gate the app opens behind.
 *
 * The property under test is that a build which cannot confirm its own approval does
 * not reach a participant. Everything else here is about saying why, because a
 * refusal nobody can act on is a support call.
 */
class ReleaseGateTest {

    private fun status(
        activated: Boolean = true,
        approved: Boolean = true,
        drifting: Boolean = false,
    ) = ReleaseStatus(
        releaseId = "rel-1",
        status = if (approved && activated) "activated" else "draft",
        activated = activated,
        approved = approved,
        usableWithParticipants = activated && approved,
        drifting = drifting,
    )

    @Test
    fun `an approved and activated release opens`() {
        assertEquals(
            GateDecision.Open,
            ReleaseGate(developmentBuild = false).evaluate(status()),
        )
    }

    @Test
    fun `an unapproved release is refused and says why`() {
        val decision = ReleaseGate(developmentBuild = false)
            .evaluate(status(approved = false, activated = false))

        assertTrue("must be blocked, was $decision", decision is GateDecision.Blocked)
        val blocked = decision as GateDecision.Blocked
        assertTrue(blocked.awaitingApproval)
        // Criterion 6 says the app says why. A refusal with no reason is not one.
        assertTrue(blocked.headline.isNotBlank())
        assertTrue(blocked.detail.isNotBlank())
    }

    @Test
    fun `an approved but not activated release is refused`() {
        // The two states are distinct on purpose. Treating them as one either lets
        // an unactivated release into the field, or blocks a perfectly good build.
        val decision = ReleaseGate(developmentBuild = false)
            .evaluate(status(approved = true, activated = false))

        assertTrue(decision is GateDecision.Blocked)
        assertTrue((decision as GateDecision.Blocked).awaitingApproval)
    }

    @Test
    fun `a drifting build is refused and names drift as the cause`() {
        val decision = ReleaseGate(developmentBuild = false)
            .evaluate(status(drifting = true))

        assertTrue(decision is GateDecision.Blocked)
        val blocked = decision as GateDecision.Blocked
        // Drift is a different conversation from approval: someone has to go and look
        // at what changed, rather than sign something.
        assertTrue(blocked.awaitingApproval.not())
        assertTrue(blocked.headline.contains("changed"))
    }

    @Test
    fun `an unreachable server denies rather than permitting`() {
        // The failure this exists to prevent is an unapproved build answering a real
        // patient's question. Not knowing must therefore mean no.
        val decision = ReleaseGate(developmentBuild = false).evaluate(ReleaseStatus.unknown())

        assertTrue("unknown must block, was $decision", decision is GateDecision.Blocked)
        assertTrue((decision as GateDecision.Blocked).headline.isNotBlank())
    }

    @Test
    fun `a development build always opens`() {
        // Not a loophole. The gate exists so no participant is given an unapproved
        // system, and nobody is a participant on a developer's own machine.
        val gate = ReleaseGate(developmentBuild = true)

        assertEquals(GateDecision.Open, gate.evaluate(status(approved = false, activated = false)))
        assertEquals(GateDecision.Open, gate.evaluate(ReleaseStatus.unknown()))
        assertEquals(GateDecision.Open, gate.evaluate(status(drifting = true)))
    }

    @Test
    fun `an unrecognised state is refused rather than treated as fine`() {
        // A server that grew a field this build does not know about must not read as
        // an approval by omission.
        val future = ReleaseStatus(
            releaseId = "rel-2",
            status = "something-new",
            activated = false,
            approved = false,
            usableWithParticipants = false,
            drifting = false,
        )

        assertTrue(ReleaseGate(developmentBuild = false).evaluate(future) is GateDecision.Blocked)
    }

    @Test
    fun `the status fields the gate reads all exist on the server view`() {
        // Pinned against the server contract by name. study_release.client_view()
        // returns exactly these keys; a rename there with no change here would leave
        // the client reading defaults and blocking every build, or worse, opening them.
        val serverKeys = setOf(
            "release_id", "status", "activated", "approved",
            "usable_with_participants", "drifting", "drift",
        )
        val clientFields = setOf(
            "releaseId", "status", "activated", "approved",
            "usableWithParticipants", "drifting", "drift",
        )

        assertEquals(serverKeys.size, clientFields.size)
        assertEquals(
            serverKeys.map { it }.sorted(),
            clientFields.map { snakeOf(it) }.sorted(),
        )
    }

    private fun snakeOf(camel: String): String {
        val out = StringBuilder()
        camel.forEachIndexed { index, c ->
            if (c.isUpperCase() && index > 0) out.append('_')
            out.append(c.lowercaseChar())
        }
        return out.toString()
    }
}