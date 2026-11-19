package org.inventcures.pallisahayak.data.reminders

import java.time.DayOfWeek
import java.time.ZonedDateTime
import java.util.UUID

/** One daily reminder as the screen lists it: seven stored rows, one a weekday. */
data class DailyReminder(
    /** What the seven rows share; removing the reminder removes them all. */
    val medicationRef: String,
    val text: String,
    /** Minutes since local midnight. 8:30 is 510. */
    val minutesOfDay: Int,
)

/**
 * The reminders kept on this phone.
 *
 * A reminder is made on the phone and stays on it: nothing here needs the server,
 * so it fires in a village with no signal (criterion 5). The table stores one row
 * a weekday, because a family schedule is weekly; a daily reminder is seven rows
 * under one medication reference.
 */
class ReminderRepository(
    private val dao: ReminderDao,
    /**
     * Called after the set of reminders changes, so the alarm that wakes the phone
     * for the next one is moved. The alarm belongs to the app module.
     */
    private val onChanged: () -> Unit = {},
    private val clock: () -> Long = System::currentTimeMillis,
    private val newId: () -> String = { UUID.randomUUID().toString() },
) {

    /** Remind every day at [minutesOfDay], saying [text]. Returns its reference. */
    suspend fun addDaily(text: String, minutesOfDay: Int, languageTag: String): String {
        require(text.isNotBlank()) { "a reminder needs something to say" }
        require(minutesOfDay in 0 until MINUTES_IN_DAY) { "minutes of day is 0 to 1439" }

        val reference = newId()
        val now = clock()
        dao.upsertAll(
            (0 until DAYS_IN_WEEK).map { day ->
                ReminderEntity(
                    id = "$reference-$day",
                    medicationRef = reference,
                    reminderText = text.trim(),
                    minutesOfDay = minutesOfDay,
                    dayOfWeek = day,
                    languageTag = languageTag,
                    enabled = true,
                    createdAt = now,
                )
            },
        )
        onChanged()
        return reference
    }

    suspend fun remove(medicationRef: String) {
        dao.deleteAllFor(medicationRef)
        onChanged()
    }

    /** The reminders in force, earliest in the day first. */
    suspend fun daily(): List<DailyReminder> =
        dao.enabled()
            .groupBy { it.medicationRef }
            .map { (reference, rows) ->
                val first = rows.first()
                DailyReminder(reference, first.reminderText, first.minutesOfDay)
            }
            .sortedBy { it.minutesOfDay }

    private companion object {
        const val DAYS_IN_WEEK = 7
        const val MINUTES_IN_DAY = 24 * 60
    }
}

/** When the phone must next wake for a reminder. Pure, so it is tested without one. */
object ReminderSchedule {

    /** [ReminderEntity.dayOfWeek] for a date: 0 is Sunday. */
    fun dayIndex(day: DayOfWeek): Int = day.value % 7

    /** Minutes since local midnight at [moment]. */
    fun minutesOfDay(moment: ZonedDateTime): Int = moment.hour * 60 + moment.minute

    /**
     * The next moment after [now], in epoch milliseconds, at which an enabled
     * reminder falls due, or null when there is none to wait for.
     *
     * Strictly after: the reminder due this very minute is being fired by the
     * alarm that asked, and waking for it again would be a loop.
     */
    fun nextTrigger(reminders: List<ReminderEntity>, now: ZonedDateTime): Long? {
        val today = dayIndex(now.dayOfWeek)
        return reminders
            .filter { it.enabled }
            .minOfOrNull { reminder ->
                val daysAhead = (reminder.dayOfWeek - today + 7) % 7
                val candidate = now.toLocalDate()
                    .plusDays(daysAhead.toLong())
                    .atTime(reminder.minutesOfDay / 60, reminder.minutesOfDay % 60)
                    .atZone(now.zone)
                val due = if (candidate.isAfter(now)) candidate else candidate.plusDays(7)
                due.toInstant().toEpochMilli()
            }
    }
}

/**
 * Which reminders have already fired on which day.
 *
 * An alarm can arrive late, and a restart or a clock change asks again, all inside
 * the window in which a reminder still counts as due. Without this the family
 * would be reminded twice, and a second reminder reads as a second dose.
 */
interface FiredReminders {
    fun wasFired(reminderId: String, epochDay: Long): Boolean
    fun markFired(reminderId: String, epochDay: Long)
}

/**
 * Finds what is due now and has not fired yet today, with no network.
 *
 * The decision of what is due is [ReminderDueResolver]'s; this adds the memory of
 * what has already been shown.
 */
class ReminderFiring(
    private val dao: ReminderDao,
    private val fired: FiredReminders,
    private val resolver: ReminderDueResolver = ReminderDueResolver(),
) {

    /** The reminders to show at [now]. Each is marked, so asking again returns none. */
    suspend fun takeDue(now: ZonedDateTime): List<DueReminder> {
        val today = now.toLocalDate().toEpochDay()
        return resolver
            .due(
                reminders = dao.enabled(),
                minutesNow = ReminderSchedule.minutesOfDay(now),
                dayOfWeek = ReminderSchedule.dayIndex(now.dayOfWeek),
            )
            .filterNot { fired.wasFired(it.id, today) }
            .onEach { fired.markFired(it.id, today) }
    }
}
