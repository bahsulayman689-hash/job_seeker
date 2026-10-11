-- 1. Companies (new in v1.5): must come before jobs reference it
create table if not exists companies (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null unique,
  pw_salt text not null,
  pw_hash text not null,
  created_at timestamptz default now()
);

-- 2. Jobs
create table if not exists jobs (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text not null,
  skills text[] not null,
  threshold int not null default 60,
  employer_email text,
  company_id uuid references companies(id) on delete cascade,
  company_name text,
  created_at timestamptz default now()
);

-- 3. Applications
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

-- 4. If the old tables already existed, add the missing columns
alter table jobs add column if not exists employer_email text;
alter table jobs add column if not exists company_id uuid references companies(id) on delete cascade;
alter table jobs add column if not exists company_name text;

-- 5. Speed
create index if not exists jobs_company_idx on jobs(company_id);
create index if not exists applications_job_idx on applications(job_id);

-- 6. Lock all tables: only the service_role key (server-side secrets) can read or write
alter table companies enable row level security;
alter table jobs enable row level security;
alter table applications enable row level security;