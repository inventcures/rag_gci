package org.inventcures.pallisahayak.data.local

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

/**
 * Household reads and writes.
 *
 * The owner check is part of every read that can reach another household, and it is
 * written into the SQL rather than applied by the caller. ADR 0006 leaves exactly
 * one boundary real, and this is it: not member versus member, but household versus
 * household. A guard that lives in a repository is a guard one call path can
 * forget, and the call path that forgets it is the one somebody probes.
 */
@Dao
interface HouseholdDao {

    /**
     * Upserts, never REPLACE.
     *
     * `OnConflictStrategy.REPLACE` is implemented as DELETE then INSERT, and the
     * foreign key on household_members is ON DELETE CASCADE. Registering or
     * re-syncing a household therefore deleted every member of it, silently, and the
     * household came back empty. `@Upsert` updates the existing row in place, so the
     * members survive. This was caught by a test that registered two members and
     * found one.
     */
    @Upsert
    suspend fun upsert(household: HouseholdEntity)

    @Upsert
    suspend fun upsertMembers(members: List<HouseholdMemberEntity>)

    @Upsert
    suspend fun upsertMember(member: HouseholdMemberEntity)

    /**
     * The caller's household, or null when they have none.
     *
     * Returns nothing rather than throwing, because "this device is not registered"
     * is an ordinary state and not an error worth handling at a call site.
     */
    @Query("SELECT * FROM households WHERE owner_id = :ownerId LIMIT 1")
    suspend fun householdFor(ownerId: String): HouseholdEntity?

    @Query("SELECT * FROM households WHERE owner_id = :ownerId LIMIT 1")
    fun observeHousehold(ownerId: String): Flow<HouseholdEntity?>

    /**
     * Members of the caller's household only.
     *
     * Joins on owner rather than accepting a household id, so a member id belonging
     * to another household cannot be reached even if one is guessed. There is
     * deliberately no `getHousehold(id: String)` overload to bypass it with.
     */
    @Query(
        """
        SELECT m.* FROM household_members m
        INNER JOIN households h ON h.id = m.household_id
        WHERE h.owner_id = :ownerId
        ORDER BY m.created_at ASC
        """
    )
    suspend fun membersFor(ownerId: String): List<HouseholdMemberEntity>

    @Query(
        """
        SELECT m.* FROM household_members m
        INNER JOIN households h ON h.id = m.household_id
        WHERE h.owner_id = :ownerId
        ORDER BY m.created_at ASC
        """
    )
    fun observeMembers(ownerId: String): Flow<List<HouseholdMemberEntity>>

    /**
     * One member, scoped to the caller's household.
     *
     * Returns null across a household boundary, which is the same answer as "no
     * such member" on purpose. Distinguishing the two would leak whether an
     * identifier exists somewhere.
     */
    @Query(
        """
        SELECT m.* FROM household_members m
        INNER JOIN households h ON h.id = m.household_id
        WHERE h.owner_id = :ownerId AND m.id = :memberId
        LIMIT 1
        """
    )
    suspend fun memberFor(ownerId: String, memberId: String): HouseholdMemberEntity?

    @Query("DELETE FROM households WHERE owner_id = :ownerId")
    suspend fun clearFor(ownerId: String)
}