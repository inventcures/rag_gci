package org.inventcures.pallisahayak.data.reminders

import android.content.Context
import org.inventcures.pallisahayak.data.local.PalliDatabase
import java.time.ZonedDateTime

/**
 * What the alarm needs from the phone's stored reminders.
 *
 * The alarm can go off when the app is not running, so it cannot count on the
 * app having been set up; it reads the database itself. That read lives here so
 * the app module, which holds the alarm, does not need the database library.
 */
object DeviceReminders {

    /** The reminders to show at [now]. Each is marked, so asking again returns none. */
    suspend fun takeDue(context: Context, now: ZonedDateTime): List<DueReminder> =
        ReminderFiring(
            PalliDatabase.get(context).reminders(),
            SharedPreferencesFiredReminders(context),
        ).takeDue(now)

    /** When the phone must next wake, in epoch milliseconds, or null for never. */
    suspend fun nextTrigger(context: Context, now: ZonedDateTime): Long? =
        ReminderSchedule.nextTrigger(PalliDatabase.get(context).reminders().enabled(), now)
}
