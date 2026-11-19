package org.inventcures.pallisahayak.reminders

import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.inventcures.pallisahayak.AppLocale
import org.inventcures.pallisahayak.MainActivity
import org.inventcures.pallisahayak.R
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.reminders.DeviceReminders
import org.inventcures.pallisahayak.data.reminders.DueReminder
import java.time.ZonedDateTime

/**
 * Wakes the phone for the next medicine reminder and shows it.
 *
 * Criterion 5: a reminder on the phone fires with no connection. The schedule and
 * the words are in the phone's database, the alarm is the phone's own, and nothing
 * here asks the server anything.
 *
 * One alarm at a time, for the next reminder due. When it goes off the receiver
 * shows what is due, then sets the alarm for the one after. The alarm is also set
 * again when the phone restarts (alarms do not survive a restart), when its clock
 * or time zone changes, when the app is updated and each time the app opens.
 *
 * The alarm is inexact on purpose. An exact alarm needs a permission Android
 * grants only to clock and calendar apps, or that the user must find in system
 * settings. An inexact one can arrive a few minutes late, more on a phone that
 * has been lying still, which is why a reminder counts as due for half an hour
 * after its time (ReminderDueResolver).
 */
object ReminderAlarms {

    private const val TAG = "PalliReminders" // not user-facing: a logcat tag
    private const val ACTION_FIRE = "org.inventcures.pallisahayak.REMINDER_DUE" // not user-facing: an intent action
    private const val CHANNEL = "medicine_reminders" // not user-facing: a notification channel id
    private const val REQUEST_ALARM = 1
    private const val REQUEST_OPEN = 2

    private val background = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Set the alarm for the next reminder, or clear it when there is none. */
    fun schedule(context: Context) {
        val app = context.applicationContext
        background.launch { reschedule(app) }
    }

    /** Show whatever is due now, then set the alarm for the next one. */
    internal suspend fun fireAndReschedule(context: Context) {
        try {
            DeviceReminders.takeDue(context, ZonedDateTime.now()).forEach { show(context, it) }
        } catch (error: RuntimeException) {
            // The next alarm must still be set, or one bad read ends every reminder.
            Log.w(TAG, "could not show the reminders due", error) // not user-facing: logcat
        }
        reschedule(context)
    }

    private suspend fun reschedule(context: Context) {
        val alarms = context.getSystemService(AlarmManager::class.java) ?: return
        val alarm = PendingIntent.getBroadcast(
            context,
            REQUEST_ALARM,
            Intent(context, ReminderReceiver::class.java).setAction(ACTION_FIRE),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val next = try {
            DeviceReminders.nextTrigger(context, ZonedDateTime.now())
        } catch (error: RuntimeException) {
            Log.w(TAG, "could not read the reminders", error) // not user-facing: logcat
            return
        }
        if (next == null) {
            alarms.cancel(alarm)
        } else {
            // Wakes the phone, and is allowed through while it is dozing.
            alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, next, alarm)
        }
    }

    private fun show(context: Context, reminder: DueReminder) {
        // In the language the reminder was made in, whatever the phone is set to.
        val localized = AppLocale.localized(context, Language.fromTag(reminder.languageTag))
        val notifications = NotificationManagerCompat.from(context)
        if (!notifications.areNotificationsEnabled()) {
            Log.w(TAG, "notifications are turned off; a reminder was not shown") // not user-facing: logcat
            return
        }
        notifications.createNotificationChannel(
            NotificationChannel(
                CHANNEL,
                localized.getString(R.string.reminders_title),
                NotificationManager.IMPORTANCE_HIGH,
            ),
        )
        val open = PendingIntent.getActivity(
            context,
            REQUEST_OPEN,
            Intent(context, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_reminder)
            .setContentTitle(localized.getString(R.string.reminder_notification_title))
            .setContentText(reminder.text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(reminder.text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_REMINDER)
            .setAutoCancel(true)
            .setContentIntent(open)
            .build()
        try {
            notifications.notify(reminder.id.hashCode(), notification)
        } catch (error: SecurityException) {
            // The permission was withdrawn between the check and here.
            Log.w(TAG, "a reminder was not shown", error) // not user-facing: logcat
        }
    }

    /**
     * Receives the alarm, and the system's notices that the alarm has to be set
     * again: a restart, a clock or time zone change, an update of the app. Every
     * one of them is handled the same way, because a reminder that fell due while
     * the phone was off or the clock moved is still due if it is inside its window.
     */
    class ReminderReceiver : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val pending = goAsync()
            val app = context.applicationContext
            background.launch {
                try {
                    fireAndReschedule(app)
                } finally {
                    pending.finish()
                }
            }
        }
    }
}
