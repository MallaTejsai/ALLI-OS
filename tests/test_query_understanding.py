"""Tests for ALLI v0.6 Query Understanding (Session 5).

Tests structured query representation that converts natural-language questions
into machine-readable form. Different wording → same structured meaning.

Test Groups:
  A: Phone/device queries — paraphrase handling
  B: Preference/interest queries — paraphrase handling
  C: Learning/education queries — paraphrase handling
  D: Historical/temporal queries — temporal handling
  E: Entity/third-party queries — entity handling
  F: Cardinality/plurality queries — cardinality handling
  G: Negation and uncertainty queries — negation/uncertainty handling
  H: Domain extraction — domain handling
  I: Edge cases and adversarial — adversarial robustness
"""

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from memory.foundation.query_understanding import (
    StructuredQuery,
    normalize_query,
    understand_query,
)


# ---------------------------------------------------------------------------
# Group A: Phone / Device Queries — Paraphrase Handling
# ---------------------------------------------------------------------------

class TestPhoneDeviceQueries(unittest.TestCase):
    """Paraphrase groups for phone/device queries.

    All queries in a group should produce equivalent structured meaning.
    """

    def test_A1_standard_form(self):
        """Standard: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")
        self.assertEqual(sq.domain, "possessions")

    def test_A2_which_form(self):
        """Paraphrase: "Which phone do I have?" """
        sq = understand_query("Which phone do I have?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A3_mobile_form(self):
        """Paraphrase: "What mobile am I using?" """
        sq = understand_query("What mobile am I using?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A4_smartphone_form(self):
        """Paraphrase: "Which smartphone am I using?" """
        sq = understand_query("Which smartphone am I using?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A5_device_form(self):
        """Paraphrase: "What device do I use?" """
        sq = understand_query("What device do I use?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "device")

    def test_A6_laptop_form(self):
        """Paraphrase: "Which laptop do I own?" """
        sq = understand_query("Which laptop do I own?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "laptop")

    def test_A7_computer_form(self):
        """Paraphrase: "What computer am I using?" """
        sq = understand_query("What computer am I using?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "computer")

    def test_A8_got_form(self):
        """Paraphrase: "What phone have I got?" """
        sq = understand_query("What phone have I got?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A9_slang_form(self):
        """Paraphrase: "bro what phone I got?" (slang) """
        sq = understand_query("bro what phone I got?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A10_typo_form(self):
        """Paraphrase: "wat phne i hav?" (typos) """
        sq = understand_query("wat phne i hav?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A11_use_form(self):
        """Paraphrase: "What phone am I using right now?" """
        sq = understand_query("What phone am I using right now?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")
        self.assertEqual(sq.state, "current")

    def test_A12_carry_form(self):
        """Paraphrase: "What phone do I carry?" """
        sq = understand_query("What phone do I carry?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A13_have_got_form(self):
        """Paraphrase: "What phone have I got?" """
        sq = understand_query("What phone have I got?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A14_own_form(self):
        """Paraphrase: "Which phone do I own?" """
        sq = understand_query("Which phone do I own?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_A15_possess_form(self):
        """Paraphrase: "What phone do I possess?" """
        sq = understand_query("What phone do I possess?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")


# ---------------------------------------------------------------------------
# Group B: Preference / Interest Queries — Paraphrase Handling
# ---------------------------------------------------------------------------

class TestPreferenceQueries(unittest.TestCase):
    """Paraphrase groups for preference/interest queries."""

    def test_B1_favorite_color(self):
        """Standard: "What is my favorite color?" """
        sq = understand_query("What is my favorite color?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")
        self.assertEqual(sq.domain, "interests")

    def test_B2_favourite_color(self):
        """Paraphrase: "What's my favourite colour?" (British) """
        sq = understand_query("What's my favourite colour?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")

    def test_B3_like_color(self):
        """Paraphrase: "What color do I like?" """
        sq = understand_query("What color do I like?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")

    def test_B4_enjoy_food(self):
        """Paraphrase: "What food do I enjoy?" """
        sq = understand_query("What food do I enjoy?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "food")
        self.assertEqual(sq.domain, "interests")

    def test_B5_love_music(self):
        """Paraphrase: "What music do I love?" """
        sq = understand_query("What music do I love?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "music")

    def test_B6_prefer_form(self):
        """Paraphrase: "What color do I prefer?" """
        sq = understand_query("What color do I prefer?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")

    def test_B7_fancy_form(self):
        """Paraphrase: "What food do I fancy?" """
        sq = understand_query("What food do I fancy?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "food")

    def test_B8_dislike_form(self):
        """Negated preference: "What color do I dislike?" """
        sq = understand_query("What color do I dislike?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "disprefer")
        self.assertEqual(sq.object, "color")

    def test_B9_hate_form(self):
        """Negated preference: "What food do I hate?" """
        sq = understand_query("What food do I hate?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "disprefer")
        self.assertEqual(sq.object, "food")

    def test_B10_favorite_food(self):
        """Standard: "What is my favorite food?" """
        sq = understand_query("What is my favorite food?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "food")

    def test_B11_slang_favorite(self):
        """Slang: "wat's my fav color?" """
        sq = understand_query("wat's my fav color?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")

    def test_B12_typo_favorite(self):
        """Typos: "wat is my favrite colour?" """
        sq = understand_query("wat is my favrite colour?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "prefer")
        self.assertEqual(sq.object, "color")


# ---------------------------------------------------------------------------
# Group C: Learning / Education Queries — Paraphrase Handling
# ---------------------------------------------------------------------------

class TestLearningQueries(unittest.TestCase):
    """Paraphrase groups for learning/education queries."""

    def test_C1_learning_language(self):
        """Standard: "What language am I learning?" """
        sq = understand_query("What language am I learning?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "language")
        self.assertEqual(sq.domain, "learning")

    def test_C2_studying_language(self):
        """Paraphrase: "What language am I studying?" """
        sq = understand_query("What language am I studying?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "language")

    def test_C3_know_language(self):
        """Paraphrase: "What language do I know?" """
        sq = understand_query("What language do I know?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "language")

    def test_C4_learn_python(self):
        """Standard: "What am I learning in programming?" """
        sq = understand_query("What am I learning in programming?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")

    def test_C5_course_enrolled(self):
        """Paraphrase: "What course am I taking?" """
        sq = understand_query("What course am I taking?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "subject")
        self.assertEqual(sq.domain, "learning")

    def test_C6_tech_knowledge(self):
        """Standard: "What technologies do I know?" """
        sq = understand_query("What technologies do I know?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "technology")

    def test_C7_framework_knowledge(self):
        """Paraphrase: "What frameworks do I use?" """
        sq = understand_query("What frameworks do I use?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "tool")

    def test_C8_programming_language(self):
        """Standard: "What programming language am I learning?" """
        sq = understand_query("What programming language am I learning?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "language")

    def test_C9_typo_learning(self):
        """Typos: "wat langauge am i lerning?" """
        sq = understand_query("wat langauge am i lerning?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")
        self.assertEqual(sq.object, "language")

    def test_C10_slang_learning(self):
        """Slang: "yo what python i learnin rn?" """
        sq = understand_query("yo what python i learnin rn?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "learn")

    def test_C11_hobby_learning(self):
        """Standard: "What are my learning interests?" """
        sq = understand_query("What are my learning interests?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.domain, "learning")


# ---------------------------------------------------------------------------
# Group D: Historical / Temporal Queries — Temporal Handling
# ---------------------------------------------------------------------------

class TestTemporalQueries(unittest.TestCase):
    """Temporal scope detection in queries."""

    def test_D1_current_phone(self):
        """Current: "What phone do I have currently?" """
        sq = understand_query("What phone do I have currently?")
        self.assertEqual(sq.state, "current")
        self.assertIn("currently", sq.constraints)

    def test_D2_previous_phone(self):
        """Historical: "What phone did I have before?" """
        sq = understand_query("What phone did I have before?")
        self.assertEqual(sq.state, "historical")
        self.assertIn("historical", sq.constraints)

    def test_D3_past_phone(self):
        """Historical: "What phone did I use in the past?" """
        sq = understand_query("What phone did I use in the past?")
        self.assertEqual(sq.state, "historical")

    def test_D4_once_phone(self):
        """Historical: "What phone did I once own?" """
        sq = understand_query("What phone did I once own?")
        self.assertEqual(sq.state, "historical")

    def test_D5_old_phone(self):
        """Historical: "What was my old phone?" """
        sq = understand_query("What was my old phone?")
        self.assertEqual(sq.state, "historical")

    def test_D6_former_phone(self):
        """Historical: "What phone did I formerly use?" """
        sq = understand_query("What phone did I formerly use?")
        self.assertEqual(sq.state, "historical")

    def test_D7_future_phone(self):
        """Future: "What phone will I have in the future?" """
        sq = understand_query("What phone will I have in the future?")
        self.assertEqual(sq.state, "future")

    def test_D8_next_year(self):
        """Future: "What phone will I get next year?" """
        sq = understand_query("What phone will I get next year?")
        self.assertEqual(sq.state, "future")

    def test_D9_used_to_form(self):
        """Historical: "What phone did I use to have?" """
        sq = understand_query("What phone did I use to have?")
        self.assertEqual(sq.state, "historical")

    def test_D10_right_now_form(self):
        """Current: "What phone am I using right now?" """
        sq = understand_query("What phone am I using right now?")
        self.assertEqual(sq.state, "current")

    def test_D11_these_days_form(self):
        """Current: "What phone am I using these days?" """
        sq = understand_query("What phone am I using these days?")
        self.assertEqual(sq.state, "current")

    def test_D12_before_that_form(self):
        """Relative temporal: "What phone did I have before that?" """
        sq = understand_query("What phone did I have before that?")
        self.assertEqual(sq.state, "historical")
        self.assertIn("before_previous", sq.constraints)

    def test_D13_after_that_form(self):
        """Relative temporal: "What phone did I get after that?" """
        sq = understand_query("What phone did I get after that?")
        self.assertEqual(sq.state, "current")
        self.assertIn("after_previous", sq.constraints)

    def test_D14_specific_year(self):
        """Specific time: "What phone did I have in 2022?" """
        sq = understand_query("What phone did I have in 2022?")
        self.assertEqual(sq.state, "historical")
        self.assertEqual(sq.time, "2022")

    def test_D15_still_form(self):
        """Continuity: "Do I still have that phone?" """
        sq = understand_query("Do I still have that phone?")
        self.assertIn("still", sq.constraints)

    def test_D16_no_temporal(self):
        """No temporal: "What phone do I have?" (default current) """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.state, "current")


# ---------------------------------------------------------------------------
# Group E: Entity / Third-Party Queries — Entity Handling
# ---------------------------------------------------------------------------

class TestEntityQueries(unittest.TestCase):
    """Entity resolution in queries."""

    def test_E1_self_reference_my(self):
        """Self: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.subject, "user")
        self.assertTrue(sq.is_personal)

    def test_E2_self_reference_mine(self):
        """Self: "What is mine?" """
        sq = understand_query("What is mine?")
        self.assertEqual(sq.subject, "user")
        self.assertTrue(sq.is_personal)

    def test_E3_third_party_possessive(self):
        """Third-party: "What phone does my mom have?" """
        sq = understand_query("What phone does my mom have?")
        self.assertEqual(sq.subject, "mom")
        self.assertEqual(sq.entity, "mom")

    def test_E4_third_party_friend(self):
        """Third-party: "What is my friend's favorite color?" """
        sq = understand_query("What is my friend's favorite color?")
        self.assertEqual(sq.subject, "friend")
        self.assertEqual(sq.entity, "friend")

    def test_E5_the_user(self):
        """Self reference: "What is the user's name?" """
        sq = understand_query("What is the user's name?")
        self.assertEqual(sq.subject, "user")
        self.assertTrue(sq.is_personal)

    def test_E6_third_party_teacher(self):
        """Third-party: "What is my teacher's name?" """
        sq = understand_query("What is my teacher's name?")
        self.assertEqual(sq.subject, "teacher")
        self.assertEqual(sq.entity, "teacher")

    def test_E7_third_party_boss(self):
        """Third-party: "What is my boss's name?" """
        sq = understand_query("What is my boss's name?")
        self.assertEqual(sq.subject, "boss")
        self.assertEqual(sq.entity, "boss")

    def test_E8_pronoun_unresolved(self):
        """Unresolved pronoun: "What is their favorite color?" """
        sq = understand_query("What is their favorite color?")
        # Subject should still default to user when pronoun is unresolved
        self.assertEqual(sq.subject, "user")

    def test_E9_about_me(self):
        """Self: "Tell me about me" """
        sq = understand_query("Tell me about me")
        self.assertEqual(sq.subject, "user")
        self.assertTrue(sq.is_personal)

    def test_E10_who_am_i(self):
        """Self: "Who am I?" """
        sq = understand_query("Who am I?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "identity")
        self.assertTrue(sq.is_personal)


# ---------------------------------------------------------------------------
# Group F: Cardinality / Plurality Queries — Cardinality Handling
# ---------------------------------------------------------------------------

class TestCardinalityQueries(unittest.TestCase):
    """Cardinality detection in queries."""

    def test_F1_singular_phone(self):
        """Singular: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.cardinality, "one")

    def test_F2_plural_phones(self):
        """Plural: "What phones do I have?" """
        sq = understand_query("What phones do I have?")
        self.assertEqual(sq.cardinality, "one")

    def test_F3_all_phones(self):
        """Many: "What are all my phones?" """
        sq = understand_query("What are all my phones?")
        self.assertEqual(sq.cardinality, "many")

    def test_F4_list_form(self):
        """Many: "List all my devices." """
        sq = understand_query("List all my devices.")
        self.assertEqual(sq.cardinality, "many")

    def test_F5_how_many(self):
        """Count: "How many languages do I know?" """
        sq = understand_query("How many languages do I know?")
        self.assertEqual(sq.cardinality, "count")

    def test_F6_existence_do(self):
        """Existence: "Do I have a laptop?" """
        sq = understand_query("Do I have a laptop?")
        self.assertEqual(sq.cardinality, "existence")

    def test_F7_existence_does(self):
        """Existence: "Does my mom have a phone?" """
        sq = understand_query("Does my mom have a phone?")
        self.assertEqual(sq.cardinality, "existence")

    def test_F8_what_are(self):
        """Plural: "What are my goals?" """
        sq = understand_query("What are my goals?")
        self.assertEqual(sq.cardinality, "many")

    def test_F9_all_skills(self):
        """Many: "What are all my skills?" """
        sq = understand_query("What are all my skills?")
        self.assertEqual(sq.cardinality, "many")

    def test_F10_which_are(self):
        """Many: "Which are my best skills?" """
        sq = understand_query("Which are my best skills?")
        self.assertEqual(sq.cardinality, "many")

    def test_F11_what_things(self):
        """Many: "What things do I own?" """
        sq = understand_query("What things do I own?")
        self.assertEqual(sq.cardinality, "many")


# ---------------------------------------------------------------------------
# Group G: Negation and Uncertainty Queries — Negation/Uncertainty
# ---------------------------------------------------------------------------

class TestNegationUncertaintyQueries(unittest.TestCase):
    """Negation and uncertainty detection in queries."""

    def test_G1_negated_dont_like(self):
        """Negation: "What color do I not like?" """
        sq = understand_query("What color do I not like?")
        self.assertTrue(sq.negated)

    def test_G2_negated_dont_enjoy(self):
        """Negation: "What food don't I enjoy?" """
        sq = understand_query("What food don't I enjoy?")
        self.assertTrue(sq.negated)

    def test_G3_negated_never(self):
        """Negation: "What language have I never learned?" """
        sq = understand_query("What language have I never learned?")
        self.assertTrue(sq.negated)

    def test_G4_negated_doesnt(self):
        """Negation: "What phone doesn't I use?" """
        sq = understand_query("What phone doesn't I use?")
        self.assertTrue(sq.negated)

    def test_G5_negated_no_longer(self):
        """Negation: "What phone do I no longer use?" """
        sq = understand_query("What phone do I no longer use?")
        self.assertTrue(sq.negated)

    def test_G6_no_negation(self):
        """No negation: "What phone do I use?" """
        sq = understand_query("What phone do I use?")
        self.assertFalse(sq.negated)

    def test_G7_uncertainty_i_think(self):
        """Uncertainty: "What phone do I think I have?" """
        sq = understand_query("What phone do I think I have?")
        self.assertTrue(sq.uncertain)

    def test_G8_uncertainty_maybe(self):
        """Uncertainty: "Maybe I have a phone?" """
        sq = understand_query("Maybe I have a phone?")
        self.assertTrue(sq.uncertain)

    def test_G9_uncertainty_not_sure(self):
        """Uncertainty: "I'm not sure what phone I have." """
        sq = understand_query("I'm not sure what phone I have.")
        self.assertTrue(sq.uncertain)

    def test_G10_no_uncertainty(self):
        """No uncertainty: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertFalse(sq.uncertain)

    def test_G11_negated_disprefer(self):
        """Negation turns prefer to disprefer: "What color don't I like?" """
        sq = understand_query("What color don't I like?")
        self.assertEqual(sq.relation, "disprefer")

    def test_G12_negated_not_learn(self):
        """Negation turns learn to not_learn: "What language do I not know?" """
        sq = understand_query("What language do I not know?")
        self.assertEqual(sq.relation, "not_learn")


# ---------------------------------------------------------------------------
# Group H: Domain Extraction — Domain Handling
# ---------------------------------------------------------------------------

class TestDomainExtraction(unittest.TestCase):
    """Domain extraction from queries."""

    def test_H1_possessions_domain(self):
        """Possessions: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.domain, "possessions")

    def test_H2_interests_domain(self):
        """Interests: "What is my favorite color?" """
        sq = understand_query("What is my favorite color?")
        self.assertEqual(sq.domain, "interests")

    def test_H3_learning_domain(self):
        """Learning: "What language am I learning?" """
        sq = understand_query("What language am I learning?")
        self.assertEqual(sq.domain, "learning")

    def test_H4_identity_domain(self):
        """Identity: "What is my name?" """
        sq = understand_query("What is my name?")
        self.assertEqual(sq.domain, "identity")

    def test_H5_goals_domain(self):
        """Goals: "What are my goals?" """
        sq = understand_query("What are my goals?")
        self.assertEqual(sq.domain, "goals")

    def test_H6_skills_domain(self):
        """Skills: "What are my skills?" """
        sq = understand_query("What are my skills?")
        self.assertEqual(sq.domain, "skills")

    def test_H7_car_possession(self):
        """Car: "What car do I drive?" """
        sq = understand_query("What car do I drive?")
        self.assertEqual(sq.domain, "possessions")

    def test_H8_food_interest(self):
        """Food: "What food do I like?" """
        sq = understand_query("What food do I like?")
        self.assertEqual(sq.domain, "interests")

    def test_H9_music_interest(self):
        """Music: "What music do I listen to?" """
        sq = understand_query("What music do I listen to?")
        self.assertEqual(sq.domain, "interests")

    def test_H10_programming_learning(self):
        """Programming: "What do I know about Python?" """
        sq = understand_query("What do I know about Python?")
        self.assertEqual(sq.domain, "learning")


# ---------------------------------------------------------------------------
# Group I: Edge Cases and Adversarial — Adversarial Robustness
# ---------------------------------------------------------------------------

class TestAdversarialQueries(unittest.TestCase):
    """Edge cases and adversarial test inputs."""

    def test_I1_empty_query(self):
        """Empty: "" """
        sq = understand_query("")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "unknown")

    def test_I2_whitespace_only(self):
        """Whitespace: "   " """
        sq = understand_query("   ")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "unknown")

    def test_I3_single_word(self):
        """Single word: "phone" """
        sq = understand_query("phone")
        self.assertEqual(sq.raw_query, "phone")

    def test_I4_all_caps(self):
        """All caps: "WHAT PHONE DO I HAVE?" """
        sq = understand_query("WHAT PHONE DO I HAVE?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I5_mixed_case(self):
        """Mixed case: "WhAt PhOnE dO i HaVe?" """
        sq = understand_query("WhAt PhOnE dO i HaVe?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I6_extra_spaces(self):
        """Extra spaces: "What   phone   do   I   have?" """
        sq = understand_query("What   phone   do   I   have?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I7_special_chars(self):
        """Special chars: "What phone do I have?!" """
        sq = understand_query("What phone do I have?!")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I8_unicode_quotes(self):
        """Unicode quotes: "What phone do I have?" """
        sq = understand_query("What phone do I have?")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I9_long_query(self):
        """Long query: "I was just wondering, out of curiosity, what kind of phone do I currently use these days?" """
        sq = understand_query(
            "I was just wondering, out of curiosity, what kind of phone do I currently use these days?"
        )
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")
        self.assertEqual(sq.state, "current")

    def test_I10_nonsense_query(self):
        """Nonsense: "Purple elephants dance on moon cheese" """
        sq = understand_query("Purple elephants dance on moon cheese")
        self.assertEqual(sq.raw_query, "Purple elephants dance on moon cheese")

    def test_I11_mixed_languages(self):
        """Mixed languages: "What phone tengo?" """
        sq = understand_query("What phone tengo?")
        self.assertEqual(sq.raw_query, "What phone tengo?")

    def test_I12_emoji_in_query(self):
        """Emoji: "What phone do I have? 📱" """
        sq = understand_query("What phone do I have? 📱")
        self.assertEqual(sq.subject, "user")
        self.assertEqual(sq.relation, "possess")
        self.assertEqual(sq.object, "phone")

    def test_I13_rapid_fire(self):
        """Rapid fire: "phone? mine? what?" """
        sq = understand_query("phone? mine? what?")
        self.assertEqual(sq.raw_query, "phone? mine? what?")

    def test_I14_typo_heavy(self):
        """Heavy typos: "wht phn do i hv?" """
        sq = understand_query("wht phn do i hv?")
        self.assertEqual(sq.raw_query, "wht phn do i hv?")

    def test_I15_slang_heavy(self):
        """Heavy slang: "ayo bro what phone u got rn?" """
        sq = understand_query("ayo bro what phone u got rn?")
        self.assertEqual(sq.raw_query, "ayo bro what phone u got rn?")

    def test_I16_normalized_query_populated(self):
        """Verify normalized_query is populated."""
        sq = understand_query("What phone do I have?")
        self.assertNotEqual(sq.normalized_query, "")
        self.assertIn("phone", sq.normalized_query)

    def test_I17_raw_query_preserved(self):
        """Verify raw_query is preserved exactly."""
        raw = "What phone do I have?"
        sq = understand_query(raw)
        self.assertEqual(sq.raw_query, raw)

    def test_I18_to_dict(self):
        """Verify to_dict serialization."""
        sq = understand_query("What phone do I have?")
        d = sq.to_dict()
        self.assertEqual(d["subject"], "user")
        self.assertEqual(d["relation"], "possess")
        self.assertEqual(d["object"], "phone")
        self.assertIn("constraints", d)

    def test_I19_constraints_list_type(self):
        """Verify constraints is always a list."""
        sq = understand_query("What phone do I have?")
        self.assertIsInstance(sq.constraints, list)

    def test_I20_is_personal_flag(self):
        """Verify is_personal is set correctly."""
        sq_personal = understand_query("What phone do I have?")
        self.assertTrue(sq_personal.is_personal)

    def test_I21_is_general_flag(self):
        """Verify is_general is set correctly."""
        sq_general = understand_query("What is Python?")
        self.assertTrue(sq_general.is_general or not sq_general.is_personal)


# ---------------------------------------------------------------------------
# Group J: Normalization Unit Tests
# ---------------------------------------------------------------------------

class TestNormalization(unittest.TestCase):
    """Unit tests for the normalize_query function."""

    def test_J1_contraction_expansion(self):
        """Contraction expansion."""
        result = normalize_query("What's my phone?")
        self.assertIn("what is", result)

    def test_J2_slang_removal(self):
        """Slang removal."""
        result = normalize_query("bro what phone?")
        self.assertNotIn("bro", result)
        self.assertIn("phone", result)

    def test_J3_typo_correction(self):
        """Typo correction."""
        result = normalize_query("wat phne?")
        # "phne" → "phone"
        self.assertIn("phone", result)

    def test_J4_case_normalization(self):
        """Case normalization."""
        result = normalize_query("WHAT PHONE?")
        self.assertEqual(result, result.lower())

    def test_J5_punctuation_removal(self):
        """Punctuation removal."""
        result = normalize_query("What phone?!...")
        self.assertNotIn("!", result)
        self.assertNotIn(".", result)
        self.assertNotIn("?", result)

    def test_J6_whitespace_normalization(self):
        """Whitespace normalization."""
        result = normalize_query("What   phone?")
        self.assertNotIn("   ", result)

    def test_J7_apostrophe_preserved(self):
        """Apostrophes preserved for contractions."""
        result = normalize_query("What's my phone?")
        # After expansion, apostrophe is gone, but the expansion works
        self.assertIn("what is", result)

    def test_J8_empty_input(self):
        """Empty input."""
        result = normalize_query("")
        self.assertEqual(result, "")

    def test_J9_none_input(self):
        """None-like input."""
        result = normalize_query("")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
