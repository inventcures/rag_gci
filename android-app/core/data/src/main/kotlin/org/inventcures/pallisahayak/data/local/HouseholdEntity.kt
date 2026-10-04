package org.inventcures.pallisahayak.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One household, the unit the record is actually shared within.
 *
 * ADR 0006 settled this: a family sharing one phone sees one record, and the Care
 * Role is a framing label rather than a privacy boundary. There is deliberately no
 * role column on this table, because a role here would invite someone to build
 * filtering on it later, and that filtering would be a promise the app cannot keep.
 * A caregiver can read the household record anyway, so a rule that appears to
 * separate them teaches a false model of the device and makes people stop checking.
 *
 * What *is* enforced is cross-household isolation. `ownerId` is the pseudonymised
 * participant the household was registered against, and it is the one boundary here
 * that is real.
 */
@Entity(
    tableName = "households",
    indices = [Index(value = ["owner_id"], unique = true)],
)
data class HouseholdEntity(
    @PrimaryKey
    val id: String,

    /**
     * Pseudonymous participant who registered this household.
     *
     * Server-generated, never a phone number or a name. This is the key that stops
     * one household reading another's record by guessing an identifier: a lookup
     * takes the caller as well as the identifier, and compares.
     */
    @ColumnInfo(name = "owner_id")
    val ownerId: String,

    @ColumnInfo(name = "site_id")
    val siteId: String,

    /** Chosen once at registration and remembered. Not re-askable per turn. */
    @ColumnInfo(name = "primary_language")
    val primaryLanguage: String,

    @ColumnInfo(name = "registered_at")
    val registeredAt: Long,

    /**
     * Server-assigned release identifier, so household records can be traced to a
     * build the same way interactions can. Protocol §7.8.
     */
    @ColumnInfo(name = "release_id")
    val releaseId: String,
)

/**
 * One person inside a household.
 *
 * Observations attach here, not to the household, or "how is she doing" has nowhere
 * to land when a family looks after three people.
 *
 * There is no phone number, no name and no ABHA number on this row. The protocol
 * treats those as server-held, and a stolen phone should not yield a roster of
 * patients. A display name is held because a family recognises it, and it is
 * treated as ordinary personal data rather than as an identifier.
 */
@Entity(
    tableName = "household_members",
    foreignKeys = [
        ForeignKey(
            entity = HouseholdEntity::class,
            parentColumns = ["id"],
            childColumns = ["household_id"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index(value = ["household_id"])],
)
data class HouseholdMemberEntity(
    @PrimaryKey
    val id: String,

    @ColumnInfo(name = "household_id")
    val householdId: String,

    /** What the family calls this person. Not an identifier. */
    @ColumnInfo(name = "display_name")
    val displayName: String,

    /**
     * Server-side patient identifier, opaque to the client.
     *
     * Kept so observations sync against the right record. Never rendered, and never
     * used as a lookup key the client supplies freely.
     */
    @ColumnInfo(name = "patient_ref")
    val patientRef: String,

    @ColumnInfo(name = "created_at")
    val createdAt: Long,
)