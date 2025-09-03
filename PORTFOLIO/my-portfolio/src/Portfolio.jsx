import React, { useMemo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Github, Linkedin, Mail, ExternalLink,
  MapPin, FileText, GraduationCap, BriefcaseBusiness,
  FolderGit2, Trophy, Award, Phone, User
} from "lucide-react";

/** ====== EDIT JUST THIS DATA BLOCK ====== */
const DATA = {
  name: "Manjunath Popuri",
  role: "Data Scientist · ML Engineer",
  location: "Binghamton, NY — Open to Relocation",
  email: "manjunathpopuri2@gmail.com",
  resumeUrl: "/Manjunath_Popuri_DS.pdf", // ensure the PDF is in /public
  heroImage: "/hero.jpeg",                // ensure the image is in /public

  socials: [
    { label: "GitHub", href: "https://github.com/Manjunath-Popuri", icon: Github },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/manjunathpopuri", icon: Linkedin },
    { label: "Email", href: "mailto:manjunathpopuri2@gmail.com", icon: Mail },
  ],

  summary:
    "Data Scientist with 3+ years across research, industry, and academia. I apply A/B testing, Bayesian inference, NLP (BERT, Transformers), and scalable ML on AWS/Azure to deliver measurable business outcomes and executive-ready insights.",

  education: [
    { school: "Binghamton University (SUNY)", degree: "M.S. in Computer Science (AI)", period: "Aug 2023 — May 2025" },
    { school: "Vasireddy Venkatadri Institute of Technology", degree: "B.E. in Computer Science & Engineering", period: "Aug 2019 — Jun 2023" },
  ],

  experience: [
    {
      company: "Binghamton University",
      title: "Research Intern (Generative AI / NLP)",
      period: "Jul 2025 — Present",
      bullets: [
        "Quantized LLMs (Mistral-7B, Phi-3) for predictive expense classification (90%+).",
        "Built pipelines (PyTorch, HF) + prescriptive models on SageMaker.",
        "Delivered dashboards for non-technical stakeholders; actionable insights."
      ],
    },
    {
      company: "Binghamton University",
      title: "Graduate Research & Teaching Assistant",
      period: "Aug 2023 — May 2025",
      bullets: [
        "Validated earnings-call NLP with A/B tests & Bayesian methods (+18% accuracy).",
        "Deployed BERT/Transformers pipelines on Azure ML at scale.",
        "Mentored 150+ students; built exec-level dashboards in Power BI/Tableau."
      ],
    },
    {
      company: "Cognizant",
      title: "Data Scientist",
      period: "Jul 2022 — Jun 2023",
      bullets: [
        "Healthcare/finance models improved accuracy by 30% (TF/PyTorch/sklearn).",
        "Cut false positives by 22% via causal inference & Bayesian optimization.",
        "ETL at scale with PySpark & SQL; CI/CD + Docker into Azure/AWS."
      ],
    },
  ],

  projects: [
    {
      name: "Personal Finance Assistant (LLM + ML)",
      description:
        "LLM-guided insights + ML expense classification with FastAPI backend and MLOps (Docker, CI/CD).",
      tags: ["LLM","FastAPI","Docker","MLOps"],
      links: [{ label: "Repo", href: "#" }, { label: "Demo", href: "#" }],
      image: "https://picsum.photos/seed/finance/1600/900"
    },
    {
      name: "Dialog Summarizer (FLAN-T5 + PEFT)",
      description:
        "DialogSum with ROUGE eval; PEFT adapters and robust training scripts.",
      tags: ["FLAN-T5","PEFT","ROUGE"],
      links: [{ label: "Paper/Repo", href: "#" }],
      image: "https://picsum.photos/seed/dialog/1600/900"
    },
    {
      name: "Toxicity-Aware Summaries (PPO + Reward Model)",
      description:
        "PPO-based detoxification using hate-speech reward model; improved safety with utility retained.",
      tags: ["RLHF","PPO","Safety"],
      links: [{ label: "Results", href: "#" }],
      image: "https://picsum.photos/seed/toxicity/1600/900"
    },
  ],

  achievements: [
    "Boosted earnings-call NLP predictive accuracy by +18% via A/B & Bayesian methods.",
    "Optimized models to reduce false positives by 22% using causal inference & Bayes opt.",
    "Mentored 150+ students; delivered executive dashboards for decision-makers."
  ],

  certifications: [
    "Azure ML (hands-on deployment experience)",
    "AWS SageMaker (end-to-end pipelines, Docker, CI/CD)",
    "Power BI / Tableau (executive dashboards)"
  ]
};
/** ====== END DATA BLOCK ====== */

const NAV_ITEMS = [
  { id: "summary", label: "Summary" },
  { id: "education", label: "Education" },
  { id: "experience", label: "Experience" },
  { id: "projects", label: "Projects" },
  { id: "achievements", label: "Achievements & Certifications" },
  { id: "contact", label: "Contact" },
];

const Section = ({ id, title, icon: Icon, children }) => (
  <section id={id} className="scroll-mt-24">
    <motion.h2
      className="mb-6 flex items-center gap-3 text-2xl font-semibold tracking-tight"
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      viewport={{ once: true }}
    >
      {Icon && <Icon className="h-6 w-6" />} {title}
    </motion.h2>
    {children}
  </section>
);

const Chip = ({ children }) => (
  <span className="inline-flex items-center rounded-full border px-3 py-1 text-xs text-zinc-700 dark:text-zinc-200">
    {children}
  </span>
);

function Navbar() {
  const [active, setActive] = useState("summary");
  useEffect(() => {
    const observers = [];
    NAV_ITEMS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => entry.isIntersecting && setActive(s.id),
        { rootMargin: "-50% 0px -50% 0px" }
      );
      obs.observe(el);
      observers.push(obs);
    });
    return () => observers.forEach((o) => o.disconnect());
  }, []);

  return (
    <nav className="sticky top-0 z-50 border-b bg-white/75 backdrop-blur dark:bg-zinc-950/70">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <a href="#top" className="flex items-center gap-2 font-semibold">
          <User className="h-5 w-5" />
          {DATA.name.split(" ")[0]}
        </a>
        <div className="hidden gap-2 sm:flex">
          {NAV_ITEMS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={`rounded-full px-3 py-1 text-sm transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                active === s.id ? "bg-zinc-900 text-white dark:bg-white dark:text-black" : ""
              }`}
            >
              {s.label.replace(" & ", " ")}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}

function ProjectCard({ p }) {
  return (
    <motion.div
      className="overflow-hidden rounded-2xl border shadow-sm transition hover:shadow-xl"
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.35 }}
    >
      <div className="relative h-44 w-full overflow-hidden sm:h-56">
        <img src={p.image} alt={p.name} className="h-full w-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
        <div className="absolute bottom-3 left-3 text-white">
          <h3 className="text-lg font-semibold">{p.name}</h3>
        </div>
      </div>
      <div className="space-y-3 p-4">
        <p className="text-sm text-zinc-700 dark:text-zinc-200">{p.description}</p>
        <div className="flex flex-wrap gap-2">
          {p.tags?.map((t) => <Chip key={t}>{t}</Chip>)}
        </div>
        <div className="flex flex-wrap gap-3">
          {p.links?.map((l) => (
            <a
              key={l.label}
              href={l.href}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-sm font-medium text-zinc-900 underline decoration-zinc-300 underline-offset-4 hover:decoration-zinc-900 dark:text-zinc-100"
            >
              {l.label} <ExternalLink className="h-4 w-4" />
            </a>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

export default function Portfolio() {
  const sections = useMemo(() => NAV_ITEMS, []);
  return (
    <div id="top" className="min-h-screen font-sans bg-white text-zinc-900 antialiased dark:bg-zinc-950 dark:text-zinc-50">
      <Navbar />

      {/* HERO */}
      <header className="mx-auto mt-6 max-w-6xl px-4">
        <div className="grid items-center gap-8 rounded-3xl bg-gradient-to-b from-white to-zinc-50 p-6 md:grid-cols-2 md:p-10">
          <motion.div
            className="order-2 md:order-1"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="mb-2 text-3xl font-extrabold tracking-tight md:text-5xl">{DATA.name}</h1>
            <div className="mb-4 flex flex-wrap items-center gap-3 text-zinc-600 dark:text-zinc-300">
              <Chip>{DATA.role}</Chip>
              <span className="inline-flex items-center gap-2 text-sm"><MapPin className="h-4 w-4" /> {DATA.location}</span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <a
                href={DATA.resumeUrl}
                className="rounded-2xl bg-zinc-900 px-4 py-2 text-sm font-medium text-white shadow transition hover:shadow-lg dark:bg-white dark:text-black"
              >
                <FileText className="mr-2 inline h-4 w-4" />
                Download Résumé
              </a>
              {DATA.socials.map((s) => (
                <a key={s.label} href={s.href} target="_blank" rel="noreferrer"
                   className="inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-sm transition hover:shadow">
                  <s.icon className="h-4 w-4" /> {s.label}
                </a>
              ))}
            </div>
          </motion.div>

          <motion.div
            className="order-1 aspect-square overflow-hidden rounded-3xl border md:order-2"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.05 }}
          >
            <img src={DATA.heroImage} alt="Manjunath Popuri" className="h-full w-full object-cover" />
          </motion.div>
        </div>
      </header>

      {/* CONTENT */}
      <main className="mx-auto max-w-6xl space-y-20 px-4 py-16">
        <Section id="summary" title="Summary" icon={User}>
          <p className="max-w-3xl text-zinc-700 dark:text-zinc-200">{DATA.summary}</p>
        </Section>

        <Section id="education" title="Education" icon={GraduationCap}>
          <div className="grid gap-4 md:grid-cols-2">
            {DATA.education.map((ed) => (
              <div key={ed.school} className="rounded-2xl border p-4">
                <div className="text-sm text-zinc-500">{ed.period}</div>
                <div className="font-medium">{ed.school}</div>
                <div className="text-zinc-700 dark:text-zinc-200">{ed.degree}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section id="experience" title="Experience" icon={BriefcaseBusiness}>
          <div className="space-y-8 border-l pl-4">
            {DATA.experience.map((e, i) => (
              <div key={i} className="relative pl-6">
                <div className="absolute left-0 top-1 h-3 w-3 rounded-full bg-zinc-900 dark:bg-white" />
                <h4 className="font-medium">{e.title} — <span className="text-zinc-600 dark:text-zinc-300">{e.company}</span></h4>
                <div className="text-xs text-zinc-500">{e.period}</div>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
                  {e.bullets.map((b, j) => <li key={j}>{b}</li>)}
                </ul>
              </div>
            ))}
          </div>
        </Section>

        <Section id="projects" title="Projects" icon={FolderGit2}>
          <div className="grid gap-6 sm:grid-cols-2">
            {DATA.projects.map((p, i) => <ProjectCard key={i} p={p} />)}
          </div>
        </Section>

        <Section id="achievements" title="Achievements & Certifications" icon={Trophy}>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-2xl border p-4">
              <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold"><Trophy className="h-5 w-5" /> Achievements</h3>
              <ul className="list-disc space-y-2 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
                {DATA.achievements.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </div>
            <div className="rounded-2xl border p-4">
              <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold"><Award className="h-5 w-5" /> Certifications</h3>
              <ul className="list-disc space-y-2 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
                {DATA.certifications.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          </div>
        </Section>

        <Section id="contact" title="Contact" icon={Phone}>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-3">
              <p className="text-zinc-700 dark:text-zinc-200">
                Want to collaborate or chat about ML/NLP? Drop a note and I’ll respond soon.
              </p>
              <div className="flex flex-wrap gap-3">
                {DATA.socials.map((s) => (
                  <a key={s.label} href={s.href} target="_blank" rel="noreferrer"
                     className="inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-sm transition hover:shadow">
                    <s.icon className="h-4 w-4" /> {s.label}
                  </a>
                ))}
              </div>
            </div>

            <form
              className="space-y-3 rounded-2xl border p-4"
              onSubmit={(e) => { e.preventDefault(); alert("Thanks! Hook this form up to EmailJS/Formspree for production."); }}
            >
              <div>
                <label className="mb-1 block text-sm">Name</label>
                <input className="w-full rounded-xl border bg-transparent px-3 py-2 outline-none focus:ring-2 focus:ring-zinc-400" required />
              </div>
              <div>
                <label className="mb-1 block text-sm">Email</label>
                <input type="email" className="w-full rounded-xl border bg-transparent px-3 py-2 outline-none focus:ring-2 focus:ring-zinc-400" required />
              </div>
              <div>
                <label className="mb-1 block text-sm">Message</label>
                <textarea className="min-h-[120px] w-full rounded-xl border bg-transparent px-3 py-2 outline-none focus:ring-2 focus:ring-zinc-400" required />
              </div>
              <button type="submit" className="w-full rounded-2xl bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:shadow-lg dark:bg-white dark:text-black">
                Send Message
              </button>
              <p className="text-xs text-zinc-500">Static form. Connect EmailJS/Formspree for production.</p>
            </form>
          </div>
        </Section>
      </main>

      <footer className="border-t bg-zinc-50/60 py-8">
        <div className="mx-auto max-w-6xl px-4 text-sm text-zinc-500">
          © {new Date().getFullYear()} {DATA.name}. Built with React + Tailwind.
        </div>
      </footer>
    </div>
  );
}