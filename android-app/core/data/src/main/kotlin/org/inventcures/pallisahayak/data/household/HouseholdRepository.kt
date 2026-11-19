package org.inventcures.pallisahayak.data.household

import org.inventcures.pallisahayak.data.local.HouseholdDao
import org.inventcures.pallisahayak.data.local.HouseholdEntity
import org.inventcures.pallisahayak.data.local.HouseholdMemberEntity
import org.inventcures.pallisahayak.data.model.Language
import java.util.UUID

/**
 * What a carer noticed about one person: which symptom, and how bad.
 *
 * Chosen from fixed lists, never typed. Free text on its way to the server's
 * patient record could carry a name or a dose, and neither belongs there.
 * [severity] is the server's scale, 0 (none) to 4 (very severe).
 */
data class Observation(
    val symptom: String,
    val severity: Int,
)

/**
 * Registering a household and recording observations against it.
 *
 * Every call takes the caller as its first argument and passes it to the DAO, which
 * joins on it. The repository does not decide who may see what, because a decision
 * made here is a decision one call path can forget. The DAO's SQL is the boundary.
 */
class HouseholdRepository(
    private val dao: HouseholdDao,
    /**
     * Where an observation is sent once its member is proven to belong to this
     * caller: the sync queue, in the app. True only when it was stored.
     *
     * A constructor dependency rather than a settable field. A mutable sink lets one
     * part of the app replace where another part's observations go, which is not a
     * capability anything here needs. Required: with no sink this used to answer
     * "saved" and keep nothing (Android review, R2).
     */
    private val observationSink: suspend (HouseholdMemberEntity, Observation) -> Boolean,
    private val clock: () -> Long = System::currentTimeMillis,
    private val newId: () -> String = { UUID.randomUUID().toString() },
) {

    /**
     * Register a household, or return the existing one.
     *
     * Idempotent on purpose. A worker who taps twice, or retries on a poor
     * connection, must not end up with two households and no way to tell which
     * holds the history.
     */
    suspend fun register(
        ownerId: String,
        siteId: String,
        language: Language,
        releaseId: String,
    ): HouseholdEntity {
        dao.householdFor(ownerId)?.let { return it }

        val household = HouseholdEntity(
            id = newId(),
            ownerId = ownerId,
            siteId = siteId,
            primaryLanguage = language.tag,
            registeredAt = clock(),
            releaseId = releaseId,
        )
        dao.upsert(household)
        return household
    }

    suspend fun addMember(
        ownerId: String,
        householdId: String,
        displayName: String,
        patientRef: String,
    ): HouseholdMemberEntity {
        require(displayName.isNotBlank()) { "a member needs a name the family recognises" }

        // Refuse rather than write: a member belonging to someone else's household
        // would be invisible in every read, and the observation would then attach to
        // nothing with no error anywhere.
        val owned = dao.householdFor(ownerId)
        check(owned?.id == householdId) {
            "household $householdId does not belong to this device"
        }

        val member = HouseholdMemberEntity(
            id = newId(),
            householdId = householdId,
            displayName = displayName.trim(),
            patientRef = patientRef,
            createdAt = clock(),
        )
        dao.upsertMember(member)
        return member
    }

    suspend fun members(ownerId: String): List<HouseholdMemberEntity> = dao.membersFor(ownerId)

    suspend fun household(ownerId: String): HouseholdEntity? = dao.householdFor(ownerId)

    /**
     * Record an observation against one member.
     *
     * The member id is resolved through [HouseholdDao.memberFor] with the caller in
     * the query, so an identifier from another household returns null rather than
     * being recorded. Silently dropping an observation would leave a gap nobody
     * notices, so this returns false and lets the caller say so.
     */
    suspend fun recordObservation(
        ownerId: String,
        memberId: String,
        observation: Observation,
    ): Boolean {
        require(observation.symptom.isNotBlank()) { "an observation cannot be empty" }
        require(observation.severity in SEVERITY_RANGE) { "severity is 0 to 4" }

        val member = dao.memberFor(ownerId, memberId) ?: return false
        return observationSink(member, observation)
    }

    private companion object {
        /** The server's SeverityLevel: none, mild, moderate, severe, very severe. */
        val SEVERITY_RANGE = 0..4
    }
}