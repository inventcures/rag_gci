package org.inventcures.pallisahayak.data.local

import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The household boundary, tested against a real database.
 *
 * ADR 0006 settles what is and is not protected. Within a household everyone reads
 * the same record, and the Care Role is a framing label rather than a boundary. The
 * one boundary that is real is between households, so that is what these tests
 * attack.
 *
 * A real Room database is used rather than a fake. The guard under test is SQL, and
 * a fake DAO would have proved nothing about the join that enforces it.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class HouseholdDaoTest {

    private lateinit var db: PalliDatabase
    private lateinit var dao: HouseholdDao

    private val ownerA = "participant-a"
    private val ownerB = "participant-b"

    @Before
    fun setUp() {
        PalliDatabase.resetForTests()
        db = androidx.room.Room
            .inMemoryDatabaseBuilder(
                ApplicationProvider.getApplicationContext(),
                PalliDatabase::class.java,
            )
            // Direct executor, because Room dispatches suspend DAO methods to its
            // transaction executor and Robolectric never resumes that continuation.
            // This is the cause of six long-misdiagnosed test failures.
            .setQueryExecutor { it.run() }
            .setTransactionExecutor { it.run() }
            .allowMainThreadQueries()
            .build()
        dao = db.households()
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    private suspend fun seedHousehold(
        householdId: String,
        ownerId: String,
        memberId: String,
        name: String,
    ) {
        dao.upsert(
            HouseholdEntity(
                id = householdId,
                ownerId = ownerId,
                siteId = "site-1",
                primaryLanguage = "hi-IN",
                registeredAt = 1_700_000_000L,
                releaseId = "rel-1",
            ),
        )
        dao.upsertMember(
            HouseholdMemberEntity(
                id = memberId,
                householdId = householdId,
                displayName = name,
                patientRef = "patient-$memberId",
                createdAt = 1_700_000_000L,
            ),
        )
    }

    @Test
    fun `a household can be registered and read back by its owner`() = runTest {
        seedHousehold("h-a", ownerA, "m-a", "Kamala")

        val household = dao.householdFor(ownerA)
        assertNotNull(household)
        assertEquals("h-a", household!!.id)
        assertEquals("hi-IN", household.primaryLanguage)
    }

    @Test
    fun `members of a household are listed together`() = runTest {
        seedHousehold("h-a", ownerA, "m-a", "Kamala")
        seedHousehold("h-a", ownerA, "m-b", "Suresh")

        val members = dao.membersFor(ownerA)
        assertEquals(2, members.size)
        assertEquals(listOf("Kamala", "Suresh"), members.map { it.displayName })
    }

    @Test
    fun `one household cannot read another household by guessing its identifier`() = runTest {
        seedHousehold("h-a", ownerA, "m-a", "Kamala")
        seedHousehold("h-b", ownerB, "m-b", "Ramesh")

        // The identifier is known, or guessed. It changes nothing.
        assertNull(dao.memberFor(ownerB, "m-a"))
        assertNull(dao.householdFor(ownerB).takeIf { it?.id == "h-a" })
        assertTrue(dao.membersFor(ownerB).none { it.id == "m-a" })
    }

    @Test
    fun `listing members returns only the caller's household`() = runTest {
        seedHousehold("h-a", ownerA, "m-a", "Kamala")
        seedHousehold("h-b", ownerB, "m-b", "Ramesh")

        assertEquals(listOf("Kamala"), dao.membersFor(ownerA).map { it.displayName })
        assertEquals(listOf("Ramesh"), dao.membersFor(ownerB).map { it.displayName })
    }

    @Test
    fun `an unregistered device has no household rather than an error`() = runTest {
        assertNull(dao.householdFor("nobody"))
        assertTrue(dao.membersFor("nobody").isEmpty())
    }

    @Test
    fun `clearing one household leaves the other intact`() = runTest {
        seedHousehold("h-a", ownerA, "m-a", "Kamala")
        seedHousehold("h-b", ownerB, "m-b", "Ramesh")

        dao.clearFor(ownerA)

        assertNull(dao.householdFor(ownerA))
        assertNotNull(dao.householdFor(ownerB))
    }
}