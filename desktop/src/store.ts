// Takes, recordings and settings, kept in the app's IndexedDB.
import type { Report } from "./engine/analysis";

const DB = "intro-coach";
let dbp: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  dbp ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore("takes");
      d.createObjectStore("audio");
      d.createObjectStore("settings");
      d.createObjectStore("files");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}

async function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const d = await db();
  return new Promise((resolve, reject) => {
    const req = fn(d.transaction(store, mode).objectStore(store));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export const get = <T>(store: string, key: string) => tx<T | undefined>(store, "readonly", (s) => s.get(key) as IDBRequest<T | undefined>);
export const put = (store: string, key: string, value: unknown) => tx(store, "readwrite", (s) => s.put(value, key));
export const del = (store: string, key: string) => tx(store, "readwrite", (s) => s.delete(key));

export interface TakeSummary { id: string; created: string; scenario: string; overall: number; duration: number; wpm: number; has_feedback: boolean }

export async function saveTake(report: Report, audio: Blob): Promise<void> {
  await put("audio", report.id!, audio);
  await put("takes", report.id!, report);
}

export async function listTakes(): Promise<TakeSummary[]> {
  const all = await tx<Report[]>("takes", "readonly", (s) => s.getAll() as IDBRequest<Report[]>);
  return all
    .map((r) => ({ id: r.id!, created: r.created!, scenario: r.scenario, overall: r.overall, duration: r.metrics?.speaking_time, wpm: r.metrics?.wpm, has_feedback: !!r.ai }))
    .sort((a, b) => (a.id < b.id ? 1 : -1));
}

export const loadTake = (id: string) => get<Report>("takes", id);
export const loadAudio = (id: string) => get<Blob>("audio", id);
export async function deleteTake(id: string): Promise<void> {
  await del("takes", id);
  await del("audio", id);
}

// ---------------------------------------------------------------- settings

export interface AiSettings { provider: string; model: string; keys: Record<string, string> }

export async function aiSettings(): Promise<AiSettings> {
  return (await get<AiSettings>("settings", "ai")) ?? { provider: "openrouter", model: "nvidia/nemotron-3.5-super-vl:free", keys: {} };
}
export const saveAiSettings = (s: AiSettings) => put("settings", "ai", s);
