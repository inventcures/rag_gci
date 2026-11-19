package org.inventcures.pallisahayak.data.local

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * A phone on an older database upgrades without losing a recorded interaction.
 *
 * The database has no destructive fallback, on purpose: a migration that does not
 * produce the schema this build expects throws when the database opens, instead
 * of deleting study data. That makes a wrong migration a crash on every upgraded
 * phone, so each one is run here first (Android review, R3).
 *
 * Each older version is built from its committed schema file, exactly as that
 * build created it, given one interaction, and then opened by this build with
 * the real migrations. Room checks the result against the schema it compiled
 * and throws if a table, column or index differs.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class MigrationTest {

    private val context: Context = ApplicationProvider.getApplicationContext()

    @Before
    fun setUp() {
        PalliDatabase.resetForTests()
        context.deleteDatabase(NAME)
    }

    @After
    fun tearDown() {
        context.deleteDatabase(NAME)
    }

    @Test
    fun `version 1 upgrades and keeps its interaction`() = upgradesFrom(1)

    @Test
    fun `version 2 upgrades and keeps its interaction`() = upgradesFrom(2)

    @Test
    fun `version 3 upgrades and keeps its interaction`() = upgradesFrom(3)

    @Test
    fun `every version before the current one has a committed schema to upgrade from`() {
        val db = open()
        val current = try {
            db.openHelper.readableDatabase.version
        } finally {
            db.close()
        }
        for (version in 1 until current) {
            assertTrue("schemas/$version.json is missing", schemaFile(version).exists())
        }
    }

    private fun upgradesFrom(version: Int) {
        createAsBuilt(version)

        val db = open()
        try {
            // Opening runs the migrations and Room's check of the result.
            val upgraded = db.openHelper.writableDatabase
            assertTrue("did not upgrade past $version", upgraded.version > version)
            upgraded.query("SELECT COUNT(*) FROM interactions WHERE id = '$KEPT'").use { rows ->
                rows.moveToFirst()
                assertEquals("the interaction recorded on version $version", 1, rows.getInt(0))
            }
        } finally {
            db.close()
        }
    }

    private fun open(): PalliDatabase =
        Room.databaseBuilder(context, PalliDatabase::class.java, NAME)
            .addMigrations(
                PalliDatabase.MIGRATION_1_2,
                PalliDatabase.MIGRATION_2_3,
                PalliDatabase.MIGRATION_3_4,
            )
            .allowMainThreadQueries()
            .build()

    /** The database as the build at [version] created it, holding one interaction. */
    private fun createAsBuilt(version: Int) {
        val schema = JSONObject(schemaFile(version).readText()).getJSONObject("database")
        val file = context.getDatabasePath(NAME)
        file.parentFile?.mkdirs()
        val raw = SQLiteDatabase.openOrCreateDatabase(file, null)

        val entities = schema.getJSONArray("entities")
        for (i in 0 until entities.length()) {
            val entity = entities.getJSONObject(i)
            val table = entity.getString("tableName")
            raw.execSQL(entity.getString("createSql").replace(TABLE_PLACEHOLDER, table))
            val indices = entity.optJSONArray("indices") ?: continue
            for (j in 0 until indices.length()) {
                raw.execSQL(indices.getJSONObject(j).getString("createSql").replace(TABLE_PLACEHOLDER, table))
            }
            if (table == "interactions") raw.execSQL(oneRow(entity))
        }
        val setup = schema.getJSONArray("setupQueries")
        for (i in 0 until setup.length()) raw.execSQL(setup.getString(i))

        raw.version = version
        raw.close()
    }

    /** An insert that fills every column of the interactions table as it was then. */
    private fun oneRow(entity: JSONObject): String {
        val fields = entity.getJSONArray("fields")
        val columns = mutableListOf<String>()
        val values = mutableListOf<String>()
        for (i in 0 until fields.length()) {
            val field = fields.getJSONObject(i)
            val column = field.getString("columnName")
            columns.add("`$column`")
            values.add(
                when {
                    column == "id" -> "'$KEPT'"
                    field.getString("affinity") == "TEXT" -> "'x'"
                    else -> "0"
                },
            )
        }
        return "INSERT INTO interactions (${columns.joinToString()}) VALUES (${values.joinToString()})"
    }

    private fun schemaFile(version: Int) = File(SCHEMAS, "$version.json")

    private companion object {
        const val NAME = "migration-test.db"
        const val KEPT = "kept-through-upgrade"
        const val TABLE_PLACEHOLDER = "\${TABLE_NAME}"

        /** Relative to the module, which is where Gradle runs its unit tests. */
        const val SCHEMAS = "schemas/org.inventcures.pallisahayak.data.local.PalliDatabase"
    }
}
