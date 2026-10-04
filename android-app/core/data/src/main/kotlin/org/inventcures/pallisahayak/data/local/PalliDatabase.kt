package org.inventcures.pallisahayak.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import org.inventcures.pallisahayak.data.reminders.ReminderDao
import org.inventcures.pallisahayak.data.reminders.ReminderEntity

/**
 * The app's on-device store.
 *
 * Two tables only. Interactions are written by the server-side recording path and
 * pushed when a connection exists; cached answers come from the offline bundle
 * built by `offline/cache_builder.py`.
 *
 * The database is deliberately small. The app holds no clinical records of its
 * own: patient observations live on the server and sync through the mobile API, so
 * a lost or stolen device does not expose a patient history.
 */
@Database(
    entities = [
        InteractionEntity::class,
        CachedAnswerEntity::class,
        HouseholdEntity::class,
        HouseholdMemberEntity::class,
        ReminderEntity::class,
    ],
    version = 3,
    exportSchema = true,
)
abstract class PalliDatabase : RoomDatabase() {

    abstract fun interactions(): InteractionDao

    abstract fun cachedAnswers(): CachedAnswerDao

    abstract fun households(): HouseholdDao

    abstract fun reminders(): ReminderDao

    companion object {
        private const val NAME = "palli-sahayak.db"

        @Volatile
        private var instance: PalliDatabase? = null

        /**
         * One database per process.
         *
         * Opening a SQLite file twice in the same process is a real source of
         * corruption, and two `PalliDatabase` objects would each hold their own
         * connection. `synchronized` on the first caller, `@Volatile` so later
         * readers on other threads see the assignment.
         */
        fun get(context: Context): PalliDatabase =
            instance ?: synchronized(this) {
                instance ?: build(context.applicationContext).also { instance = it }
            }

        private fun build(context: Context): PalliDatabase =
            // Deliberately no fallbackToDestructiveMigration.
            //
            // Version 1 has not shipped, so nothing needs migrating yet. But if a
            // later version adds a column and this builder silently drops the
            // tables on upgrade, it deletes the recorded interactions that the
            // study protocol depends on, and the loss is invisible until someone
            // counts rows six weeks later. Without a fallback, a version mismatch
            // throws at open time instead, which is a loud failure in front of the
            // developer who caused it. That is the right trade for study data.
            Room.databaseBuilder(context, PalliDatabase::class.java, NAME)
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
                .build()

        /**
         * Adds the reminders table, same reasoning as 1 to 2.
         *
         * Additive and written out, because the alternative is deleting the recorded
         * interactions the study depends on.
         */
        val MIGRATION_2_3 = object : androidx.room.migration.Migration(2, 3) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS reminders (" +
                        "id TEXT NOT NULL, medication_ref TEXT NOT NULL, " +
                        "reminder_text TEXT NOT NULL, minutes_of_day INTEGER NOT NULL, " +
                        "day_of_week INTEGER NOT NULL, language_tag TEXT NOT NULL, " +
                        "enabled INTEGER NOT NULL, created_at INTEGER NOT NULL, " +
                        "PRIMARY KEY(id))",
                )
            }
        }

        /**
         * Adds the household tables without touching study data already stored.
         *
         * Written out rather than inferred, because the only alternative is deleting
         * recorded interactions, and that is not a trade worth making for
         * convenience. The change is additive: existing rows are untouched, so a
         * device that upgrades keeps its history and the study keeps its evidence.
         */
        val MIGRATION_1_2 = object : androidx.room.migration.Migration(1, 2) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS households (" +
                        "id TEXT NOT NULL, owner_id TEXT NOT NULL, site_id TEXT NOT NULL, " +
                        "primary_language TEXT NOT NULL, registered_at INTEGER NOT NULL, " +
                        "release_id TEXT NOT NULL, PRIMARY KEY(id))",
                )
                db.execSQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS index_households_owner_id " +
                        "ON households (owner_id)",
                )
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS household_members (" +
                        "id TEXT NOT NULL, household_id TEXT NOT NULL, " +
                        "display_name TEXT NOT NULL, patient_ref TEXT NOT NULL, " +
                        "created_at INTEGER NOT NULL, PRIMARY KEY(id), " +
                        "FOREIGN KEY(household_id) REFERENCES households(id) " +
                        "ON DELETE CASCADE ON UPDATE NO ACTION)",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_household_members_household_id " +
                        "ON household_members (household_id)",
                )
            }
        }

        /** Test seam. A test cannot share the process-wide instance above. */
        internal fun resetForTests() {
            instance?.close()
            instance = null
        }
    }
}
