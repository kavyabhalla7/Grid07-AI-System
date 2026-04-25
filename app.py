import streamlit as st
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from phase1_router import route_post_to_bots
from phase2_engine import generate_post
from phase3_combat import generate_defense_reply, ThreadMessage

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(page_title="Grid07 AI", layout="wide")

st.title("Grid07 — AI Cognitive Routing System")
st.markdown("---")

# ──────────────────────────────────────────────
# Mode Selector
# ──────────────────────────────────────────────
mode = st.radio(
    "Choose Mode:",
    ["Content Generation", "Combat Mode"]
)

# ──────────────────────────────────────────────
# MODE 1 — CONTENT GENERATION
# ──────────────────────────────────────────────
if mode == "Content Generation":

    st.subheader("Generate AI Posts")

    user_input = st.text_area("Enter a post:", height=120)

    if st.button("Run System"):

        if not user_input:
            st.warning("Please enter a post")
        else:

            # ───────── Phase 1 ─────────
            st.subheader("🔹 Phase 1 — Routing")

            matches = route_post_to_bots(user_input)

            if not matches:
                st.error("❌ No bots matched")
            else:
                for m in matches:
                    st.write(f"✅ {m.bot_name} → similarity: {m.similarity:.2f}")

            # ───────── Phase 2 ─────────
            st.subheader("🔹 Phase 2 — Generated Posts")

            for m in matches:
                result = generate_post(m.bot_id)

                st.markdown(f"### 🤖 {m.bot_name}")
                st.markdown(f"💬 {result['post_content']}")


# ──────────────────────────────────────────────
# MODE 2 — COMBAT ENGINE
# ──────────────────────────────────────────────
elif mode == "Combat Mode":

    st.subheader("AI Debate / Combat Engine")

    parent_post = st.text_area(
        "Original Post:",
        "Electric Vehicles are a complete scam. The batteries degrade in 3 years."
    )

    user_reply = st.text_area(
        "Your Reply to the Bot:",
        "Show me one peer-reviewed study backing those numbers."
    )

    if st.button("Generate Combat Reply"):

        # Sample conversation history
        comment_history = [
            ThreadMessage(
                author="bot_a",
                content="That is statistically false. Modern EV batteries retain 90% capacity after 100,000 miles."
            ),
            ThreadMessage(
                author="human",
                content="You're just repeating corporate propaganda."
            ),
        ]

        result = generate_defense_reply(
            bot_id="bot_a",
            parent_post=parent_post,
            comment_history=comment_history,
            human_reply=user_reply
        )

        st.markdown("### 🤖 Bot Reply")
        st.markdown(f"💬 {result.reply_content}")

        if result.injection_detected:
            st.error(f"🚨 Injection Detected: {result.injection_reason}")
        else:
            st.success("✅ No injection detected")