package org.inventcures.pallisahayak.data.household

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.reminders.FiredReminders
import org.inventcures.pallisahayak.data.reminders.ReminderFiring
import org.inventcures.pallisahayak.data.reminders.ReminderRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
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
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * The household screen's behaviour, against a real database (Android review, R2).
 *
 * Before this the household, observation and reminder code had no screen: an
 * observation with nowhere to go was reported as saved and dropped, and no
 * reminder ever fired. These pin what the screen now promises a participant.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class HouseholdViewModelTest {

    private lateinit var db: PalliDatabase

    /** What reached the sync queue: the person's reference and the observation. */
    private val queued = mutableListOf<Pair<String, Observation>>()
    private var queueWorks = true
    private var alarmMoves = 0
    private var ids = 0

    private val session = SessionContext(
        participantId = "participant-a",
        careRole = "family_caregiver",
        siteId = "CCHRC",
        language = "hi-IN",
        releaseId = "study-1@abc1234",
        releaseApproved = true,
    )

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
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    private fun reminders() = ReminderRepository(
        dao = db.reminders(),
        onChanged = { alarmMoves++ },
        newId = { "reminder-${ids++}" },
    )

    private fun TestScope.viewModel() = HouseholdViewModel(
        households = HouseholdRepository(
            dao = db.households(),
            observationSink = { member, observation ->
                if (queueWorks) queued += member.patientRef to observation
                queueWorks
            },
        ),
        reminders = reminders(),
        session = session,
        scope = CoroutineScope(UnconfinedTestDispatcher(testScheduler)),
        newPatientRef = { "patient-${ids++}" },
    )

    private fun TestScope.withKamala(): HouseholdViewModel {
        val vm = viewModel()
        vm.addMember("Kamala")
        testScheduler.advanceUntilIdle()
        return vm
    }

    @Test
    fun `adding a person lists them and makes the household once`() = runTest {
        val vm = withKamala()
        vm.addMember("Suresh")
        testScheduler.advanceUntilIdle()

        assertEquals(listOf("Kamala", "Suresh"), vm.state.value.members.map { it.displayName })
        assertEquals(1, vm.state.value.members.map { it.householdId }.distinct().size)
        assertTrue(vm.state.value.loaded)
    }

    @Test
    fun `a blank name adds nobody`() = runTest {
        val vm = viewModel()
        vm.addMember("   ")
        testScheduler.advanceUntilIdle()

        assertTrue(vm.state.value.members.isEmpty())
    }

    @Test
    fun `an observation is queued under the person's reference, never their name`() = runTest {
        val vm = withKamala()
        val kamala = vm.state.value.members.single()

        vm.recordObservation(kamala.id, Symptom.PAIN, Severity.SEVERE)
        testScheduler.advanceUntilIdle()

        assertEquals(HouseholdNotice.OBSERVATION_SAVED, vm.state.value.notice)
        assertEquals(listOf(kamala.patientRef to Observation("pain", 3)), queued)
        assertFalse(queued.single().first.contains("Kamala"))
    }

    @Test
    fun `an observation that could not be queued is reported as not saved`() = runTest {
        val vm = withKamala()
        queueWorks = false

        vm.recordObservation(vm.state.value.members.single().id, Symptom.NAUSEA, Severity.MILD)
        testScheduler.advanceUntilIdle()

        assertEquals(HouseholdNotice.OBSERVATION_FAILED, vm.state.value.notice)
        assertTrue(queued.isEmpty())
    }

    @Test
    fun `an observation against someone not in this household is reported as not saved`() = runTest {
        val vm = withKamala()

        vm.recordObservation("someone-else", Symptom.PAIN, Severity.MILD)
        testScheduler.advanceUntilIdle()

        assertEquals(HouseholdNotice.OBSERVATION_FAILED, vm.state.value.notice)
        assertTrue(queued.isEmpty())
    }

    @Test
    fun `a reminder is kept for every day and the alarm is moved`() = runTest {
        val vm = withKamala()

        vm.addReminder("Morphine", "Morphine for Kamala", minutesOfDay = 8 * 60)
        testScheduler.advanceUntilIdle()

        assertEquals(HouseholdNotice.REMINDER_SAVED, vm.state.value.notice)
        val reminder = vm.state.value.reminders.single()
        assertEquals("Morphine for Kamala", reminder.text)
        assertEquals(480, reminder.minutesOfDay)
        assertEquals(7, db.reminders().enabled().size)
        assertEquals((0..6).toList(), db.reminders().enabled().map { it.dayOfWeek }.sorted())
        assertEquals("hi-IN", db.reminders().enabled().first().languageTag)
        assertEquals(1, alarmMoves)
    }

    @Test
    fun `a medicine name that carries an amount is refused and nothing is kept`() = runTest {
        val vm = withKamala()

        vm.addReminder("Morphine 10 mg", "Morphine 10 mg for Kamala", minutesOfDay = 8 * 60)
        testScheduler.advanceUntilIdle()

        assertEquals(HouseholdNotice.REMINDER_HAS_DOSE, vm.state.value.notice)
        assertTrue(vm.state.value.reminders.isEmpty())
        assertTrue(db.reminders().enabled().isEmpty())
        assertEquals(0, alarmMoves)
    }

    @Test
    fun `removing a reminder removes every day of it`() = runTest {
        val vm = withKamala()
        vm.addReminder("Morphine", "Morphine for Kamala", minutesOfDay = 8 * 60)
        testScheduler.advanceUntilIdle()

        vm.removeReminder(vm.state.value.reminders.single().medicationRef)
        testScheduler.advanceUntilIdle()

        assertTrue(vm.state.value.reminders.isEmpty())
        assertTrue(db.reminders().enabled().isEmpty())
        assertNull(vm.state.value.notice)
        assertEquals(2, alarmMoves)
    }

    @Test
    fun `a reminder fires once in its window and not again that day`() = runTest {
        reminders().addDaily("Morphine for Kamala", minutesOfDay = 8 * 60, languageTag = "hi-IN")
        val firing = ReminderFiring(db.reminders(), InMemoryFired())
        val india = ZoneId.of("Asia/Kolkata")
        fun at(day: Int, hour: Int, minute: Int) = ZonedDateTime.of(2026, 11, day, hour, minute, 0, 0, india)

        assertTrue("before its time", firing.takeDue(at(18, 7, 59)).isEmpty())
        // The alarm is inexact: ten minutes late is still shown.
        assertEquals(listOf("Morphine for Kamala"), firing.takeDue(at(18, 8, 10)).map { it.text })
        assertTrue("a second alarm in the window", firing.takeDue(at(18, 8, 20)).isEmpty())
        assertTrue("past its window", firing.takeDue(at(18, 9, 0)).isEmpty())
        assertEquals("the next day", 1, firing.takeDue(at(19, 8, 0)).size)
    }
}

private class InMemoryFired : FiredReminders {
    private val days = mutableMapOf<String, Long>()
    override fun wasFired(reminderId: String, epochDay: Long): Boolean = days[reminderId] == epochDay
    override fun markFired(reminderId: String, epochDay: Long) {
        days[reminderId] = epochDay
    }
}
