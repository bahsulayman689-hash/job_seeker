-- Run this once in Supabase: SQL Editor -> New query -> Run
create table if not exists jobs (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text not null,
  skills text[] not null,
  threshold int not null default 60,
  created_at timestamptz default now()
);

create table if not exists applications (
  id uuid primary key default gen_random_uuid(),
  job_id uuid references jobs(id) on delete cascade,
  name text not null,
  contact text not null,
  score int not null,
  found text[],
  missing text[],
  status text not null default 'new',
  created_at timestamptz default now(),
  unique (job_id, contact)
);

-- Lock the tables: with RLS on and no policies, only the service_role key
-- (kept in Streamlit secrets, server-side) can read or write.
alter table jobs enable row level security;
alter table applications enable row level security;

-- Add after the first run: per-job notification email
alter table jobs add column if not exists employer_email text;