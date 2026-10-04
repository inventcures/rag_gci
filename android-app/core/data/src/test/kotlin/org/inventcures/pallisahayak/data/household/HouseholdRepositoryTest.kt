package org.inventcures.pallisahayak.data.household

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.local.HouseholdEntity
import org.inventcures.pallisahayak.data.local.HouseholdMemberEntity
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.model.Language
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Household registration and observation routing, against a real database.
 *
 * The property that matters is that an observation lands on the right person, and
 * that it cannot land on someone in another household. Everything else here is
 * supporting detail for that.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class HouseholdRepositoryTest {

    private lateinit var db: PalliDatabase
    private lateinit var repo: HouseholdRepository
    private val sent = mutableListOf<Pair<String, String>>()
    private var counter = 0

    private val ownerA = "participant-a"
    private val ownerB = "participant-b"

    @Before
    fun setUp() {
        PalliDatabase.resetForTests()
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            PalliDatabase::class.java,
        )
            .setQueryExecutor { it.run() }
            .setTransactionExecutor { it.run() }
            .allowMainThreadQueries()
            .build()

        repo = HouseholdRepository(
            dao = db.households(),
            observationSink = { member, observation ->
                sent += member.id to observation
                true
            },
            clock = { 1_700_000_000L },
            newId = { "id-${counter++}" },
        )
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    @Test
    fun `registering a household stores the chosen language`() = runTest {
        val household = repo.register(ownerA, "site-1", Language.MARATHI, "rel-1")

        assertEquals("mr-IN", household.primaryLanguage)
        assertEquals(household.id, repo.household(ownerA)?.id)
    }

    @Test
    fun `registering twice returns the same household rather than a second one`() = runTest {
        val first = repo.register(ownerA, "site-1", Language.MARATHI, "rel-1")
        val second = repo.register(ownerA, "site-1", Language.MARATHI, "rel-1")

        assertEquals(first.id, second.id)
        assertEquals(1, repo.household(ownerA)?.let { 1 } ?: 0)
    }

    @Test
    fun `an observation lands on the member it was recorded against`() = runTest {
        val household = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")
        val kamala = repo.addMember(ownerA, household.id, "Kamala", "p-1")
        repo.addMember(ownerA, household.id, "Suresh", "p-2")

        assertTrue(repo.recordObservation(ownerA, kamala.id, "Slept better overnight"))

        assertEquals(listOf(kamala.id to "Slept better overnight"), sent)
    }

    @Test
    fun `an observation against another household's member is refused`() = runTest {
        val hA = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")
        val kamala = repo.addMember(ownerA, hA.id, "Kamala", "p-1")
        repo.register(ownerB, "site-1", Language.HINDI, "rel-1")

        // The identifier is known. It still changes nothing.
        assertFalse(repo.recordObservation(ownerB, kamala.id, "Guessed write"))
        assertTrue("a refused observation reached the sink", sent.isEmpty())
    }

    @Test
    fun `a member cannot be added to a household this device does not own`() = runTest {
        val hA = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")

        val failure = runCatching {
            repo.addMember(ownerB, hA.id, "Intruder", "p-9")
        }

        assertTrue(failure.isFailure)
    }

    @Test
    fun `a member needs a name the family will recognise`() = runTest {
        val hA = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")

        assertTrue(runCatching { repo.addMember(ownerA, hA.id, "   ", "p-1") }.isFailure)
    }

    @Test
    fun `an empty observation is refused rather than recorded as content`() = runTest {
        val hA = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")
        val member = repo.addMember(ownerA, hA.id, "Kamala", "p-1")

        assertTrue(runCatching { repo.recordObservation(ownerA, member.id, "  ") }.isFailure)
        assertTrue(sent.isEmpty())
    }

    @Test
    fun `every member of a household is visible to everyone in it`() = runTest {
        val hA = repo.register(ownerA, "site-1", Language.HINDI, "rel-1")
        repo.addMember(ownerA, hA.id, "Kamala", "p-1")
        repo.addMember(ownerA, hA.id, "Suresh", "p-2")
        repo.addMember(ownerA, hA.id, "Meena", "p-3")

        // One record, read by whoever holds the phone. ADR 0006: no filtering here.
        assertEquals(listOf("Kamala", "Suresh", "Meena"), repo.members(ownerA).map { it.displayName })
    }

    @Test
    fun `an unregistered device has no household rather than an error`() = runTest {
        assertNull(repo.household("nobody"))
        assertTrue(repo.members("nobody").isEmpty())
    }
}