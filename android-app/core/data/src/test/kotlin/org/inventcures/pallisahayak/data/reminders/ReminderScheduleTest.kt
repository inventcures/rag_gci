package org.inventcures.pallisahayak.data.reminders

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * When the phone next wakes for a reminder (Android review, R2).
 *
 * The alarm is set for whatever this returns, so a wrong answer is a reminder
 * that never fires, with nothing on screen to say so.
 */
class ReminderScheduleTest {

    private val india = ZoneId.of("Asia/Kolkata")

    /** Wednesday 18 November 2026, 10:00 in India. Wednesday is day 3. */
    private val now = ZonedDateTime.of(2026, 11, 18, 10, 0, 0, 0, india)

    private fun reminder(day: Int, minutes: Int, enabled: Boolean = true) = ReminderEntity(
        id = "r-$day-$minutes",
        medicationRef = "m",
        reminderText = "Medicine for Kamala",
        minutesOfDay = minutes,
        dayOfWeek = day,
        languageTag = "hi-IN",
        enabled = enabled,
        createdAt = 0,
    )

    private fun at(day: Int, hour: Int, minute: Int): Long =
        ZonedDateTime.of(2026, 11, day, hour, minute, 0, 0, india).toInstant().toEpochMilli()

    @Test
    fun `no reminders means nothing to wake for`() {
        assertNull(ReminderSchedule.nextTrigger(emptyList(), now))
        assertNull(ReminderSchedule.nextTrigger(listOf(reminder(3, 20 * 60, enabled = false)), now))
    }

    @Test
    fun `a reminder later today is next`() {
        assertEquals(at(18, 20, 30), ReminderSchedule.nextTrigger(listOf(reminder(3, 20 * 60 + 30)), now))
    }

    @Test
    fun `a reminder earlier today waits for the same day next week`() {
        assertEquals(at(25, 8, 0), ReminderSchedule.nextTrigger(listOf(reminder(3, 8 * 60)), now))
    }

    @Test
    fun `the reminder due this very minute is not next, the one after it is`() {
        val next = ReminderSchedule.nextTrigger(listOf(reminder(3, 10 * 60), reminder(4, 10 * 60)), now)

        assertEquals(at(19, 10, 0), next)
    }

    @Test
    fun `the earliest of several is chosen, across the end of the week`() {
        // Sunday is day 0: from a Wednesday it is four days ahead, not three behind.
        val sunday = reminder(0, 7 * 60)
        val friday = reminder(5, 21 * 60)

        assertEquals(at(20, 21, 0), ReminderSchedule.nextTrigger(listOf(sunday, friday), now))
        assertEquals(at(22, 7, 0), ReminderSchedule.nextTrigger(listOf(sunday), now))
    }

    @Test
    fun `the day index is the one the due resolver uses`() {
        assertEquals(0, ReminderSchedule.dayIndex(java.time.DayOfWeek.SUNDAY))
        assertEquals(3, ReminderSchedule.dayIndex(now.dayOfWeek))
        assertEquals(600, ReminderSchedule.minutesOfDay(now))
    }
}
