"""The four situations: what's asked, how long to talk, the shape that works, and what the checklist looks for.

Each `checks` entry names a detector in analysis.py (`key`) and says how it's graded and coached in this situation.
Check weights add up to 90; the remaining 10 points of the structure score are for fitting the target length.
"""

SCENARIOS = {
    "interview": {
        "label": "Interview",
        "prompt": "Tell me about yourself.",
        "target": 75,
        "shape": {
            "title": "Present → Past → Proof → Future",
            "steps": [
                {"name": "Present", "say": "Your name and what you do now, in one plain sentence.",
                 "example": "I'm Alex, a product designer at a healthcare startup, where I lead our mobile app."},
                {"name": "Past", "say": "The background that makes you credible for this role.",
                 "example": "Before that I spent five years designing tools for nurses and clinic staff."},
                {"name": "Proof", "say": "One concrete result, ideally with a number.",
                 "example": "My last redesign cut appointment booking time by 40%."},
                {"name": "Future", "say": "Why this role, and what you want to do next.",
                 "example": "Now I want to design for millions of patients, which is why this role stood out."},
            ],
            "land": "End on the Future line with a falling pitch, then stop. Don't add \"so yeah\".",
        },
        "checks": [
            {"key": "name", "weight": 15, "label": "Your name, in the first sentence",
             "fix_title": "Say your name in the first sentence", "fix": "\"Hi, I'm ___.\" Interviewers anchor on it."},
            {"key": "role", "weight": 20, "label": "What you do now",
             "fix_title": "Say what you do now, plainly", "fix": "One sentence: \"I'm a ___ at ___ / I lead ___.\""},
            {"key": "experience", "weight": 10, "label": "Your background (years, places)",
             "fix_title": "Anchor your experience", "fix": "Give a time frame or context: \"for five years in healthcare products\"."},
            {"key": "proof", "weight": 25, "label": "A concrete result, with a number",
             "fix_title": "Add one concrete result with a number",
             "fix": "\"I cut booking time by 40%\" sticks; \"I worked on lots of projects\" doesn't."},
            {"key": "close", "weight": 20, "label": "Why this role / what's next",
             "fix_title": "End with where you're heading", "fix": "\"Now I'm looking to ___, which is why I'm excited about this role.\""},
        ],
    },
    "networking": {
        "label": "Networking",
        "prompt": "So, what do you do?",
        "target": 30,
        "shape": {
            "title": "Who → Who you help → Hook",
            "steps": [
                {"name": "Who", "say": "Your name and a one-line role a stranger understands. No jargon.",
                 "example": "I'm Sam, I'm an accountant."},
                {"name": "Who you help", "say": "What you do for whom, with one concrete example.",
                 "example": "I help small restaurants understand their numbers. Last month one café found $800 a month in forgotten subscriptions."},
                {"name": "Hook", "say": "Something you're working on now, or a question back to keep it a conversation.",
                 "example": "Right now I'm building a one-page monthly checklist for owners. What brings you here?"},
            ],
            "land": "Hand the conversation back with a question. Thirty seconds, then listen.",
        },
        "checks": [
            {"key": "name", "weight": 20, "label": "Your name, up front",
             "fix_title": "Lead with your name", "fix": "\"Hi, I'm ___.\" before anything else."},
            {"key": "role", "weight": 20, "label": "A role a stranger understands",
             "fix_title": "Say what you do in plain words", "fix": "\"I'm a software engineer\" beats a list of technologies."},
            {"key": "value", "weight": 25, "label": "Who you help, with an example",
             "fix_title": "Say who you help", "fix": "\"I help ___ do ___\", then one concrete example people can picture."},
            {"key": "hook", "weight": 25, "label": "A hook or a question back",
             "fix_title": "End with a hook or a question", "fix": "\"Right now I'm working on ___. What about you?\" keeps the conversation going."},
        ],
    },
    "meeting": {
        "label": "Team meeting",
        "prompt": "Let's go around and introduce ourselves.",
        "target": 25,
        "shape": {
            "title": "Name → Role here → Current focus → How to reach you",
            "steps": [
                {"name": "Name", "say": "Your name, said slowly.", "example": "Hi everyone, I'm Jordan."},
                {"name": "Role here", "say": "Your team and role in this group's terms.",
                 "example": "I just joined the data team as an analyst."},
                {"name": "Current focus", "say": "What you're working on this month or quarter.",
                 "example": "This quarter I'm focused on our customer churn dashboard."},
                {"name": "How to reach you", "say": "When people should come to you, or one human detail.",
                 "example": "If you need numbers on anything customer-related, ping me."},
            ],
            "land": "Keep it under 30 seconds and finish on the invitation, not on a trailing \"...yeah\".",
        },
        "checks": [
            {"key": "name", "weight": 20, "label": "Your name",
             "fix_title": "Say your name clearly", "fix": "\"Hi everyone, I'm ___.\""},
            {"key": "role", "weight": 25, "label": "Your team and role",
             "fix_title": "Say your team and role", "fix": "\"I'm on the ___ team as a ___.\""},
            {"key": "current_work", "weight": 25, "label": "What you're working on now",
             "fix_title": "Say what you're working on now", "fix": "\"This quarter I'm focused on ___.\" gives people a reason to talk to you."},
            {"key": "reach", "weight": 20, "label": "When to come to you (or a human detail)",
             "fix_title": "Tell people when to come to you", "fix": "\"If you need help with ___, ping me.\" Or one detail outside work."},
        ],
    },
    "talk": {
        "label": "Talk / podcast",
        "prompt": "Before we start, tell the audience who you are.",
        "target": 45,
        "shape": {
            "title": "Name → Credibility → Why it matters to them → What's coming",
            "steps": [
                {"name": "Name", "say": "Your name and role, briefly.", "example": "I'm Priya, I run customer support at a software company."},
                {"name": "Credibility", "say": "One result or experience that earns their attention.",
                 "example": "Last year we cut our average reply time from 24 hours to under one."},
                {"name": "Why it matters to them", "say": "Connect it to the audience's problem.",
                 "example": "If you've ever kept a customer waiting, this one's for you."},
                {"name": "What's coming", "say": "Preview what they'll get from the next minutes.",
                 "example": "Today I'll show you the three changes that made the difference."},
            ],
            "land": "Finish on the preview, with energy, and go straight into the content.",
        },
        "checks": [
            {"key": "name", "weight": 15, "label": "Your name and role",
             "fix_title": "Open with your name", "fix": "\"I'm ___, I ___.\" Short; the audience wants the content."},
            {"key": "credibility", "weight": 25, "label": "Why they should listen to you",
             "fix_title": "Give one reason to listen to you", "fix": "One result or experience: \"We've ___ for ___ customers.\""},
            {"key": "audience", "weight": 25, "label": "Why it matters to the audience",
             "fix_title": "Connect it to the audience", "fix": "\"If you've ever ___, this is for you.\""},
            {"key": "preview", "weight": 25, "label": "What's coming",
             "fix_title": "Preview what's coming", "fix": "\"Today I'll show you ___.\" gives them a reason to keep listening."},
        ],
    },
}
