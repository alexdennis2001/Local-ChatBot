# Local-ChatBot
This is a local chatbot solution with a database in the cloud using supabase


<h3>1. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>2. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>3. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>4. Create accounts</h3>

- Create a free account on Supabase: https://supabase.com/
- Create an API key for OpenAI: https://platform.openai.com/api-keys

<h3>6. Execute SQL queries in Supabase</h3>

Execute the following SQL query in Supabase:

```
-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1536) -- 1536 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;
```

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python ingest_in_db.py
streamlit run agentic_rag_streamlit.py
```