# Nota — Minimal Notes App

A beautiful, minimalist notes app with Google Auth, multiple users, rich editing, and aesthetic export templates.

## Features

- **Google OAuth** — Sign in with Google, each user has their own notes
- **Rich Notes Editor** — Markdown content, emoji, color themes per note
- **Smart Organization** — Pin notes, archive, trash, full-text search
- **Tags** — Create colored tags, filter by tag
- **Export Templates** — 5 aesthetic templates (Minimal, Warm Journal, Dark Editorial, Pastel Soft, Elegant Paper)
- **Export Formats** — PDF, PNG, Plain Text, Markdown
- **Mobile-first** — Fully responsive, works great on phone + desktop
- **Real-time Autosave** — Saves as you type with debounce

---

## Tech Stack

- **Next.js 14** (App Router)
- **NextAuth.js** — Google OAuth + JWT sessions
- **Prisma** — ORM
- **PostgreSQL** — via Neon (free tier)
- **Tailwind CSS** — styling
- **Lora + DM Sans** — beautiful font pairing

---

## Local Setup

### 1. Clone & Install

```bash
git clone <your-repo>
cd notes-app
npm install
```

### 2. Set up Database (Neon — free)

1. Go to [neon.tech](https://neon.tech) and create a free account
2. Create a new project
3. Copy the **Connection String** (format: `postgresql://...`)

### 3. Set up Google OAuth

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or use existing)
3. Go to **APIs & Services → OAuth consent screen**
   - Choose "External"
   - Fill in app name, email
   - Add scope: `email`, `profile`
4. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   - Application type: **Web application**
   - Authorized redirect URIs: `http://localhost:3000/api/auth/callback/google`
5. Copy **Client ID** and **Client Secret**

### 4. Environment Variables

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

Fill in your values:

```env
DATABASE_URL="postgresql://your-neon-connection-string"
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="run: openssl rand -base64 32"
GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET="your-client-secret"
```

Generate NEXTAUTH_SECRET:
```bash
openssl rand -base64 32
```

### 5. Push Database Schema

```bash
npm run db:push
```

### 6. Run Dev Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Deploy to Vercel

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/notes-app
git push -u origin main
```

### 2. Import to Vercel

1. Go to [vercel.com](https://vercel.com) → New Project
2. Import your GitHub repo
3. Vercel auto-detects Next.js — no config needed

### 3. Add Environment Variables in Vercel

In your Vercel project → Settings → Environment Variables, add:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | Your Neon connection string |
| `NEXTAUTH_URL` | `https://your-vercel-domain.vercel.app` |
| `NEXTAUTH_SECRET` | Your generated secret |
| `GOOGLE_CLIENT_ID` | Your Google client ID |
| `GOOGLE_CLIENT_SECRET` | Your Google client secret |

> ⚠️ **Important**: Use the Neon connection string with `?sslmode=require` appended if SSL errors occur.

### 4. Update Google OAuth Redirect URI

In Google Cloud Console → Your OAuth Client → Authorized redirect URIs, add:
```
https://your-vercel-domain.vercel.app/api/auth/callback/google
```

### 5. Deploy!

Vercel will auto-deploy on every push to `main`. The `vercel.json` runs `prisma generate && next build` automatically.

---

## Project Structure

```
notes-app/
├── app/
│   ├── api/
│   │   ├── auth/[...nextauth]/  # NextAuth routes
│   │   ├── notes/               # CRUD API
│   │   └── tags/                # Tags API
│   ├── (app)/dashboard/         # Main app page
│   ├── globals.css
│   └── layout.tsx
├── components/
│   ├── notes/
│   │   ├── DashboardLayout.tsx  # Main layout
│   │   ├── Sidebar.tsx          # Navigation + tags
│   │   ├── NotesList.tsx        # Notes list panel
│   │   ├── NoteEditor.tsx       # Writing interface
│   │   └── MobileNav.tsx        # Mobile bottom nav
│   ├── export/
│   │   └── ExportModal.tsx      # Export with templates
│   ├── LoginPage.tsx
│   └── Providers.tsx
├── lib/
│   └── prisma.ts
├── prisma/
│   └── schema.prisma
├── types/
│   └── index.ts
└── vercel.json
```

---

## Export Templates

| Template | Style |
|----------|-------|
| **Minimal** | Clean white, generous space, green accent |
| **Warm Journal** | Cream gradient, warm brown tones |
| **Dark Editorial** | Dark charcoal, sage green accent |
| **Pastel Soft** | Lavender to rose gradient |
| **Elegant Paper** | Warm beige, left gold border accent |

---

## Database Schema

- **User** — NextAuth user (Google profile)
- **Note** — title, content, emoji, color, pinned/archived/trashed
- **Tag** — named, colored, per-user
- **NoteTag** — many-to-many join

---

## License

MIT
