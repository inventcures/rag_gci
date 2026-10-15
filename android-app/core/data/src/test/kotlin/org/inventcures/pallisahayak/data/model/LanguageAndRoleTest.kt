package org.inventcures.pallisahayak.data.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The language and role vocabulary, checked against the code that has to agree with it.
 *
 * One test reads a file outside the Kotlin source tree on purpose. The client
 * list and the server's Supported Language table must hold the same eleven
 * languages under the same codes: a request, a bundle or a string resource under
 * another spelling ("od-IN" for Odia) misses. Asserting the count is not enough,
 * so the codes and tags themselves are compared.
 */
class LanguageAndRoleTest {

    /** Repository root, found by walking up from the module directory. */
    private fun repoRoot(): File {
        var dir = File(System.getProperty("user.dir") ?: ".")
        while (!File(dir, "supported_languages.py").exists()) {
            dir = dir.parentFile ?: error("could not locate supported_languages.py")
        }
        return dir
    }

    /** The Supported Language table's (code, tag) pairs, read from the server's source. */
    private fun tableLanguages(): Set<Pair<String, String>> {
        val source = File(repoRoot(), "supported_languages.py").readText()
        return Regex("SupportedLanguage\\(\\s*\"([a-z]{2})\", \"([a-z]{2}-IN)\"")
            .findAll(source)
            .map { it.groupValues[1] to it.groupValues[2] }
            .toSet()
            .also { check(it.size == 11) { "expected the eleven table rows, found $it" } }
    }

    @Test
    fun `the client supports exactly the Supported Language table's eleven, by its codes`() {
        // Odia is "or-IN" in the table. The app sent "od-IN" until ticket 24.
        assertEquals(tableLanguages(), Language.ALL.map { it.code to it.tag }.toSet())
    }

    @Test
    fun `an Odia tag stored by an older build still resolves to Odia`() {
        assertEquals(Language.ODIA, Language.fromTag("od-IN"))
        assertEquals(Language.ODIA, Language.fromTag("od"))
        assertEquals(Language.ODIA, Language.fromTag("or"))
        assertEquals("or-IN", Language.fromTag("od-IN")?.tag)
    }

    @Test
    fun `every language carries an endonym so a worker finds their own`() {
        // A translated name is what an English-first build would offer, and it is
        // exactly the case where a worker cannot find their own language.
        Language.ALL.forEach { assertTrue("${it.tag} has no endonym", it.endonym.isNotBlank()) }
        assertEquals(Language.ALL.size, Language.ALL.map { it.endonym }.toSet().size)
    }

    @Test
    fun `a supported tag resolves`() {
        assertEquals(Language.MARATHI, Language.fromTag("mr-IN"))
        assertEquals(Language.MARATHI, Language.fromTag("MR-in"))
        assertEquals(Language.MARATHI, Language.fromTag("mr"))
        assertEquals(Language.MARATHI, Language.fromTag("  mr-IN  "))
    }

    @Test
    fun `an unsupported language resolves to nothing rather than a substitute`() {
        // The criterion is explicit: refused plainly, never silently substituted.
        // Defaulting here would answer a Gujarati worker in English while the screen
        // said Gujarati, and they would believe they had been understood.
        assertNull(Language.fromTag("fr-FR"))
        assertNull(Language.fromTag(""))
        assertNull(Language.fromTag(null))
        assertFalse(Language.isSupported("de-DE"))
    }

    @Test
    fun `an unsupported language never resolves to English by accident`() {
        // Guards the specific failure of a prefix match: "en-GB" is not English-India,
        // and treating it as supported would be a substitution.
        assertNull(Language.fromTag("en-GB"))
        assertNull(Language.fromTag("en-US"))
    }

    @Test
    fun `an unknown role falls back to the common case rather than failing`() {
        // Deliberately unlike language. A role is an analysis attribute and exposes
        // nothing, so continuing as the ASHA Worker is safe. A wrong language would
        // mislead the user about being understood.
        assertEquals(CareRole.ASHA_WORKER, CareRole.fromWire("someone_new"))
        assertEquals(CareRole.PATIENT, CareRole.fromWire("patient"))
        assertEquals(CareRole.ASHA_WORKER, CareRole.fromWire(null))
    }

    @Test
    fun `the three role icons differ in silhouette, not only in detail`() {
        // ADR 0006: three similar human figures are indistinguishable at small sizes
        // to someone who cannot read the label, so the glyphs must differ in outline.
        val icons = CareRole.ALL.map { it.iconKey }
        assertEquals(3, icons.toSet().size)
    }

    @Test
    fun `no role presents itself with a lock or a shield`() {
        // The switch must never look like a privacy control. Any glyph in this set
        // would imply a boundary ADR 0006 says does not exist.
        val forbidden = listOf("lock", "shield", "key", "fingerprint", "verified_user", "security")
        CareRole.ALL.forEach { role ->
            val icon = role.iconKey.lowercase()
            forbidden.forEach { glyph ->
                assertFalse(
                    "${role.wireName} uses $glyph, which implies access control",
                    icon.contains(glyph),
                )
            }
        }
    }

    @Test
    fun `role wire names are distinct and stable`() {
        val names = CareRole.ALL.map { it.wireName }
        assertEquals(3, names.toSet().size)
        // These go into the study log. Renaming one silently splits a cohort.
        assertTrue(names.contains("asha_worker"))
        assertNotEquals(names[0], names[1])
    }
}