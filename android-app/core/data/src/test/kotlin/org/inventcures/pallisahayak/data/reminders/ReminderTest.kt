package org.inventcures.pallisahayak.data.reminders

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Medication reminders, and the claim that they fire offline.
 *
 * Criterion 5. The property that matters is that nothing here consults the network,
 * so the test asserts the stored text is enough: a reminder created while online
 * still resolves and speaks its own words with the connection gone.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ReminderTest {

    private lateinit var db: PalliDatabase
    private lateinit var resolver: ReminderDueResolver

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
        resolver = ReminderDueResolver()
    }

    @After
    fun tearDown() {
        db.close()
        PalliDatabase.resetForTests()
    }

    private fun reminder(
        id: String = "r1",
        at: Int = 8 * 60 + 30,
        day: Int = 2,
        enabled: Boolean = true,
        text: String = "It is time for her medicine. Follow the advice your doctor gave you.",
    ) = ReminderEntity(
        id = id,
        medicationRef = "med-1",
        reminderText = text,
        minutesOfDay = at,
        dayOfWeek = day,
        languageTag = "hi-IN",
        enabled = enabled,
        createdAt = 1_700_000_000L,
    )

    // -- criterion 5 --------------------------------------------------------

    @Test
    fun `a stored reminder is due without any network`() {
        val due = resolver.due(listOf(reminder(at = 510, day = 2)), minutesNow = 515, dayOfWeek = 2)

        assertEquals(1, due.size)
        // The text is on the row, which is the whole reason this works offline.
        assertEquals(
            "It is time for her medicine. Follow the advice your doctor gave you.",
            due.single().text,
        )
    }

    @Test
    fun `a reminder survives being stored and read back`() = runTest {
        db.reminders().upsert(reminder(id = "r1", at = 510, day = 2))
        db.reminders().upsert(reminder(id = "r2", at = 570, day = 2))

        val stored = db.reminders().enabled()

        assertEquals(2, stored.size)
        assertEquals(8 * 60 + 30, stored.first { it.id == "r1" }.minutesOfDay)
    }

    @Test
    fun `a reminder stored online still resolves with the connection gone`() = runTest {
        // Everything needed is on the device: the schedule and the words. No field on
        // ReminderEntity is a server reference that has to be re-fetched to speak.
        db.reminders().upsert(reminder(id = "r1", at = 510, day = 2))

        val stored = db.reminders().enabled()
        val due = resolver.due(stored, minutesNow = 512, dayOfWeek = 2)

        assertEquals(1, due.size)
        assertTrue("text must be on the row", due.single().text.isNotBlank())
    }

    // -- the delivery window ------------------------------------------------

    @Test
    fun `a reminder does not fire early`() {
        assertTrue(resolver.due(listOf(reminder(at = 510)), minutesNow = 505, dayOfWeek = 2).isEmpty())
    }

    @Test
    fun `a delayed reminder still fires rather than being dropped`() {
        // The failure that matters: a device asleep at 8:30 wakes at 8:50 and the
        // reminder is silently lost, so the family never knows a dose was missed.
        val due = resolver.due(
            listOf(reminder(at = 510)),
            minutesNow = 540,
            dayOfWeek = 2,
            windowMinutes = 30,
        )

        assertEquals(1, due.size)
    }

    @Test
    fun `a reminder outside the window has already passed`() {
        assertTrue(resolver.due(listOf(reminder(at = 510)), minutesNow = 570, dayOfWeek = 2).isEmpty())
    }

    @Test
    fun `a reminder fires only on its own day`() {
        assertTrue(resolver.due(listOf(reminder(at = 510, day = 2)), minutesNow = 515, dayOfWeek = 3).isEmpty())
    }

    @Test
    fun `a disabled reminder never fires`() {
        val disabled = reminder(at = 510, day = 2, enabled = false)
        assertTrue(resolver.due(listOf(disabled), minutesNow = 515, dayOfWeek = 2).isEmpty())
    }

    @Test
    fun `due reminders come back in schedule order`() {
        val due = resolver.due(
            // Both inside the window, so the ordering is what is being asserted. An
            // earlier version of this test put one 95 minutes out, which correctly
            // did not fire, and so asserted nothing about ordering at all.
            listOf(reminder(id = "late", at = 600, day = 2), reminder(id = "early", at = 580, day = 2)),
            minutesNow = 605,
            dayOfWeek = 2,
        )

        assertEquals(listOf("early", "late"), due.map { it.id })
    }

    @Test
    fun `a reminder can be turned off without being deleted`() = runTest {
        db.reminders().upsert(reminder(id = "r1", at = 510, day = 2))
        db.reminders().setEnabled("r1", false)

        assertTrue(db.reminders().enabled().isEmpty())
        // Still there, so it can be turned back on. Deleting would lose the schedule.
        assertTrue(db.reminders().find("r1") != null)
    }

    @Test
    fun `an empty schedule resolves to nothing rather than failing`() {
        assertTrue(resolver.due(emptyList(), minutesNow = 515, dayOfWeek = 2).isEmpty())
    }

    @Test
    fun `the enabled stream reflects a stored reminder`() = runTest {
        db.reminders().upsert(reminder(id = "r1"))
        assertTrue(db.reminders().observeEnabled().first().isNotEmpty())
        assertFalse(db.reminders().observeEnabled().first().any { !it.enabled })
    }
}