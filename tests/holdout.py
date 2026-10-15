"""A second batch of labelled intros, written after the detectors were tuned on corpus.py, to measure how well the
checks generalise to phrasings they weren't fitted to."""

HOLDOUT = [
    {"scenario": "interview", "note": "sales, quota",
     "text": "Hello, I'm Fatima. I'm an account executive selling software to hospitals. I've been in sales for nine years. Last quarter I closed 140 percent of my quota. I'd love to bring that energy to your enterprise team.",
     "expect": {"name": True, "role": True, "experience": True, "proof": True, "close": True}},
    {"scenario": "interview", "note": "chef, no numbers",
     "text": "I'm Luca, a sous chef at an Italian restaurant in Boston. I trained in Florence and Milan. I run the kitchen on busy weekend nights. My goal is to open my own place one day.",
     "expect": {"name": True, "role": True, "experience": True, "proof": False, "close": True}},
    {"scenario": "interview", "note": "manager, money saved, no close",
     "text": "My name is Victor. I manage IT for a school district. In the past three years we moved 40 schools to the cloud and saved 200,000 dollars.",
     "expect": {"name": True, "role": True, "experience": True, "proof": True, "close": False}},
    {"scenario": "interview", "note": "nervous, no structure",
     "text": "Um, hi. So I guess I should talk about myself. I'm pretty hardworking and I like learning new things.",
     "expect": {"name": False, "role": False, "experience": False, "proof": False, "close": False}},
    {"scenario": "interview", "note": "researcher, publication count",
     "text": "Hi, I'm Dr. Mei Chen. I'm a postdoctoral researcher in neuroscience. I've published 12 papers on memory over the last five years. Next I want to lead my own lab.",
     "expect": {"name": True, "role": True, "experience": True, "proof": True, "close": True}},
    {"scenario": "interview", "note": "support agent, rating",
     "text": "I'm Kwame, and I work in customer support for an airline. I've handled over 20,000 calls with a 98 percent satisfaction rating. I'm applying because I want to move into training new agents.",
     "expect": {"name": True, "role": True, "experience": False, "proof": True, "close": True}},

    {"scenario": "networking", "note": "real estate",
     "text": "Hi, I'm Dana. I'm a real estate agent. I help first-time buyers find homes they can actually afford. Are you looking at all, or just browsing?",
     "expect": {"name": True, "role": True, "value": True, "hook": True}},
    {"scenario": "networking", "note": "developer, no value",
     "text": "I'm Igor. I'm a developer. I mostly write Go.",
     "expect": {"name": True, "role": True, "value": False, "hook": False}},
    {"scenario": "networking", "note": "nonprofit, 'we give'",
     "text": "Nice to meet you, I'm Bea. I run a nonprofit that gives laptops to kids in rural schools. Lately I've been looking for corporate partners.",
     "expect": {"name": True, "role": True, "value": True, "hook": True}},
    {"scenario": "networking", "note": "no name, question",
     "text": "I'm a career coach. I help mid-career people switch into tech without starting over. What do you do?",
     "expect": {"name": False, "role": True, "value": True, "hook": True}},
    {"scenario": "networking", "note": "artist",
     "text": "Hey, I'm Tomás, I'm an illustrator. I draw children's books, mostly for small publishers. I'm working on my first graphic novel right now.",
     "expect": {"name": True, "role": True, "value": True, "hook": True}},
    {"scenario": "networking", "note": "only name",
     "text": "Hi, I'm Zoe. Great event, right?",
     "expect": {"name": True, "role": False, "value": False, "hook": True}},

    {"scenario": "meeting", "note": "new hire, ask me",
     "text": "Hi team, I'm Priya. I just started as a backend engineer on billing. This month I'm getting to know the codebase. Ask me anything about Rust.",
     "expect": {"name": True, "role": True, "current_work": True, "reach": True}},
    {"scenario": "meeting", "note": "manager, no current work",
     "text": "Hello everyone, I'm Greg, the new head of sales for Europe. Please reach out if you want to shadow a client call.",
     "expect": {"name": True, "role": True, "current_work": False, "reach": True}},
    {"scenario": "meeting", "note": "hobby detail",
     "text": "I'm Yuki, data engineer. I'm migrating our pipelines to Airflow. On weekends I bake way too much bread.",
     "expect": {"name": True, "role": True, "current_work": True, "reach": True}},
    {"scenario": "meeting", "note": "no role",
     "text": "Hey, I'm Owen. Excited to be here. Looking forward to working with you all.",
     "expect": {"name": True, "role": False, "current_work": False, "reach": False}},
    {"scenario": "meeting", "note": "designer, slack",
     "text": "Hi all, I'm Nadia and I'm the designer on the checkout squad. We're redesigning the cart page right now. You can always find me on Slack.",
     "expect": {"name": True, "role": True, "current_work": True, "reach": True}},
    {"scenario": "meeting", "note": "contractor",
     "text": "Good morning. I'm Paul, I'm a contractor helping with the security audit. I'll be here for six weeks.",
     "expect": {"name": True, "role": True, "current_work": True, "reach": False}},

    {"scenario": "talk", "note": "doctor, audience question",
     "text": "I'm Dr. Ravi Shah, an emergency physician. I've seen thousands of sports injuries. How many of you run at least once a week? Today I'll explain the three mistakes that hurt runners most.",
     "expect": {"name": True, "credibility": True, "audience": True, "preview": True}},
    {"scenario": "talk", "note": "no preview",
     "text": "Hi, I'm Jen. I've coached high school debate for fifteen years. If you've ever frozen in front of a crowd, you know the feeling.",
     "expect": {"name": True, "credibility": True, "audience": True, "preview": False}},
    {"scenario": "talk", "note": "podcast, 'we're going to dig into'",
     "text": "Thanks for having me, I'm Carlos. I founded two food delivery startups. We're going to dig into why most of them fail.",
     "expect": {"name": True, "credibility": True, "audience": False, "preview": True}},
    {"scenario": "talk", "note": "no credibility, no name",
     "text": "Good afternoon. Have you ever felt like your inbox runs your life? In this talk, I'll share a simple email routine.",
     "expect": {"name": False, "credibility": False, "audience": True, "preview": True}},
    {"scenario": "talk", "note": "engineer, outage",
     "text": "I'm Helga, a site reliability engineer at a bank. Last year we cut our outages by 60 percent. If you're on call, this one is for you. I'm going to walk you through our playbook.",
     "expect": {"name": True, "credibility": True, "audience": True, "preview": True}},
    {"scenario": "talk", "note": "name only, filler",
     "text": "Hello, my name is Ivan. Okay, so, yeah, let's begin.",
     "expect": {"name": True, "credibility": False, "audience": False, "preview": False}},
]

# Written after both the Python and the TypeScript checks were tuned; both scored 64/68 (94%) before these were fixed.
FRESH = [
 {
  "scenario": "interview",
  "note": "teacher to L&D",
  "text": "Good afternoon, I'm Rachel. I'm a learning designer at an insurance company. I taught middle school math for eight years before that. My onboarding course cut new hire ramp-up from six weeks to three. I'm looking for a role where I can design learning at a bigger scale.",
  "expect": {
   "name": True,
   "role": True,
   "experience": True,
   "proof": True,
   "close": True
  }
 },
 {
  "scenario": "interview",
  "note": "no proof, no close",
  "text": "Hi, my name is Ade. I'm a civil engineer. I've worked on bridges and roads for about six years.",
  "expect": {
   "name": True,
   "role": True,
   "experience": True,
   "proof": False,
   "close": False
  }
 },
 {
  "scenario": "interview",
  "note": "no role stated",
  "text": "Hello, I'm Sven. I've always loved building things. Over the past four years I shipped three mobile games with over a million downloads. I'd love to bring that to your studio.",
  "expect": {
   "name": True,
   "role": False,
   "experience": True,
   "proof": True,
   "close": True
  }
 },
 {
  "scenario": "interview",
  "note": "rambling",
  "text": "So, yeah, I guess I'm kind of a generalist. I've done stuff in a lot of areas. I'm a hard worker.",
  "expect": {
   "name": False,
   "role": False,
   "experience": False,
   "proof": False,
   "close": False
  }
 },
 {
  "scenario": "networking",
  "note": "vet",
  "text": "Hi, I'm Noor, I'm a vet. I help dog owners keep their older dogs active. What do you do?",
  "expect": {
   "name": True,
   "role": True,
   "value": True,
   "hook": True
  }
 },
 {
  "scenario": "networking",
  "note": "no value",
  "text": "I'm Pete. I'm in software. Nice event.",
  "expect": {
   "name": True,
   "role": False,
   "value": False,
   "hook": False
  }
 },
 {
  "scenario": "networking",
  "note": "architect, current project",
  "text": "Hey, I'm In\u00e9s, an architect. I design homes for families who want to live greener. At the moment I'm working on a straw-bale house.",
  "expect": {
   "name": True,
   "role": True,
   "value": True,
   "hook": True
  }
 },
 {
  "scenario": "networking",
  "note": "no name",
  "text": "I run a bike repair shop. We fix commuter bikes the same day so people aren't stuck. Do you cycle?",
  "expect": {
   "name": False,
   "role": True,
   "value": True,
   "hook": True
  }
 },
 {
  "scenario": "meeting",
  "note": "complete",
  "text": "Hi all, I'm Leo, the new QA lead. Right now I'm setting up our test plan for the checkout launch. If something breaks, tell me first.",
  "expect": {
   "name": True,
   "role": True,
   "current_work": True,
   "reach": True
  }
 },
 {
  "scenario": "meeting",
  "note": "no reach",
  "text": "Hello, I'm Mina from marketing. This week I'm planning the spring campaign.",
  "expect": {
   "name": True,
   "role": True,
   "current_work": True,
   "reach": False
  }
 },
 {
  "scenario": "meeting",
  "note": "just name",
  "text": "Hi, I'm Josh. Glad to be here.",
  "expect": {
   "name": True,
   "role": False,
   "current_work": False,
   "reach": False
  }
 },
 {
  "scenario": "meeting",
  "note": "human detail",
  "text": "I'm Carla, analyst on the risk team. I'm rebuilding our fraud model this quarter. Fun fact, I used to be a pastry chef.",
  "expect": {
   "name": True,
   "role": True,
   "current_work": True,
   "reach": True
  }
 },
 {
  "scenario": "talk",
  "note": "complete",
  "text": "I'm Dr. Omar Haddad, a sleep researcher. I've studied sleep in over 5,000 shift workers. If you've ever worked a night shift, you know the fog. Tonight I'll show you how to clear it.",
  "expect": {
   "name": True,
   "credibility": True,
   "audience": True,
   "preview": True
  }
 },
 {
  "scenario": "talk",
  "note": "no audience",
  "text": "Hi, I'm Grace. I lead the design team at a bank. We'll look at three redesigns today.",
  "expect": {
   "name": True,
   "credibility": True,
   "audience": False,
   "preview": True
  }
 },
 {
  "scenario": "talk",
  "note": "no preview, no credibility",
  "text": "Hello, I'm Tim. Have you ever lost your keys? It happens to all of us.",
  "expect": {
   "name": True,
   "credibility": False,
   "audience": True,
   "preview": False
  }
 },
 {
  "scenario": "talk",
  "note": "podcast, no name",
  "text": "Thanks for having me on. I've spent twenty years teaching kids to code. Let's talk about why most coding classes fail.",
  "expect": {
   "name": False,
   "credibility": True,
   "audience": False,
   "preview": True
  }
 }
]
