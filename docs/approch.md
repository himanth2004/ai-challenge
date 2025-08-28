

# Idea at a Glance

What We Built\
A fast, privacy-first check to tell if someone really feels human.
People move their mouse with tiny hesitations, uneven speeds, and
imperfect paths. Bots? They tend to move smoothly, repeat patterns, or
act superhumanly fast. By watching two natural signals---mouse movements
and typing patterns---we catch imposters in real time, without sending
sensitive data anywhere. The whole thing runs live on Render with a
dashboard that shows clear, instant results.

# What We Want to Do

• Detect bots, fake users, and spoofing in real time.\
• Use behavior signals that are hard to fake but easy to understand.\
• Keep models small, privacy-aware, and easy to run anywhere.

# The Two Signals

• Mouse Dynamics:\
Real people make little 'wiggles', curved paths, and speed changes. Bots
usually draw straighter lines at a steady pace with less variation.\
\
• Typing Patterns:\
Humans type with rhythm---pauses, bursts, and some inconsistency. Bots
either paste text ridiculously fast or produce overly uniform patterns.\
Because these two are pretty independent, it's much harder for bots to
fake both perfectly.

# How It Works, Step by Step

1\. Front end\
We capture small windows of mouse coordinates as you move, and take the
text you type (but without storing raw keystrokes).

2\. Feature extraction\
For mouse: calculate simple stats like average position, spread,
distances between steps---things that show how natural or robotic the
motion is.\
For typing: clean the text, convert to TF--IDF features, and estimate
typing speed. If speed goes above 250 WPM, we flag it immediately as a
bot.

3\. Models\
Mouse movements get classified by a Random Forest.\
Typing gets analyzed by Logistic Regression on TF--IDF features.\
Both models are light and fast, saved as .pkl files for easy loading.

4\. Real time predictions\
The app sends small amounts of data to backend endpoints and gets quick
responses. The dashboard displays whether each signal looks human or
bot, with confidence and timing info.

5\. Clear feedback\
You see both mouse and typing results separately, plus a detailed
activity log that shows what's happening behind the scenes.

# Why This Approach Works

• Bots struggle to copy tiny unpredictable motions and human typing
rhythm at once.\
• The features we use are simple and interpretable --- variance, path
length, word frequency, typing speed --- so it's easy to tune and
understand why the system decides something's bot-like.\
• Small models mean the system works fast and can run anywhere, from
cloud servers to desktops or phones.\
• All data used is aggregated or cleaned, so we never store or send
sensitive personal info.

# What Makes Our Model Different

• We check two separate behaviors, so a bot must fool both, not just
one.\
• Superhuman typing speed instantly triggers a bot flag before even
calling the server.\
• Clear and honest UI logs let anyone see the system's decisions and
timings in real time.\
• Designed with privacy first, and built to easily add new signals later
(like touch or motion).\
• It's already live and working, but ready for future upgrades too.

# To Sum Up

This is a straightforward way to capture the "human feel" of
interaction---how people move and type---and make smart, explainable
decisions fast. The dual check, simple rules, and privacy focus make it
reliable now and flexible for whatever comes next.
