const DB_NAME = "hope-textbook-workspaces";
const DB_VERSION = 1;
const PROJECT_STORE = "projects";
const PAGE_STORE = "pages";

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("브라우저 저장소 요청에 실패했습니다."));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error || new Error("브라우저 저장소 저장에 실패했습니다."));
    transaction.onabort = () => reject(transaction.error || new Error("브라우저 저장소 저장이 취소되었습니다."));
  });
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in globalThis)) {
      reject(new Error("현재 브라우저에서는 작업 저장 기능을 사용할 수 없습니다."));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(PROJECT_STORE)) {
        const projects = database.createObjectStore(PROJECT_STORE, { keyPath: "id" });
        projects.createIndex("updatedAt", "updatedAt");
      }
      if (!database.objectStoreNames.contains(PAGE_STORE)) {
        const pages = database.createObjectStore(PAGE_STORE, { keyPath: "id" });
        pages.createIndex("projectId", "projectId");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("브라우저 저장소를 열 수 없습니다."));
  });
}

export async function listTextbookProjects() {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readonly");
    const records = await requestResult(transaction.objectStore(PROJECT_STORE).getAll());
    return records.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  } finally {
    database.close();
  }
}

export async function getTextbookProject(projectId) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readonly");
    return await requestResult(transaction.objectStore(PROJECT_STORE).get(projectId));
  } finally {
    database.close();
  }
}

export async function createTextbookProject(file, pageCount) {
  const now = new Date().toISOString();
  const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const record = {
    id,
    name: file.name,
    fileName: file.name,
    fileType: file.type || "application/pdf",
    fileSize: file.size,
    fileLastModified: file.lastModified,
    pdfBlob: file.slice(0, file.size, file.type || "application/pdf"),
    pageCount,
    createdAt: now,
    updatedAt: now,
  };

  const database = await openDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readwrite");
    const done = transactionDone(transaction);
    transaction.objectStore(PROJECT_STORE).put(record);
    await done;
    return record;
  } finally {
    database.close();
  }
}

export async function listSavedPages(projectId) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(PAGE_STORE, "readonly");
    const index = transaction.objectStore(PAGE_STORE).index("projectId");
    const records = await requestResult(index.getAll(IDBKeyRange.only(projectId)));
    return records.sort((a, b) => a.pageNumber - b.pageNumber);
  } finally {
    database.close();
  }
}

export async function saveWorkspacePage(projectId, pageNumber, result, settings) {
  const now = new Date().toISOString();
  const pageRecord = {
    id: `${projectId}:${pageNumber}`,
    projectId,
    pageNumber,
    result,
    settings,
    reviewStatus: result?.page_description?.review_status || "unreviewed",
    savedAt: now,
  };

  const database = await openDatabase();
  try {
    const transaction = database.transaction([PAGE_STORE, PROJECT_STORE], "readwrite");
    const done = transactionDone(transaction);
    transaction.objectStore(PAGE_STORE).put(pageRecord);
    const projects = transaction.objectStore(PROJECT_STORE);
    const project = await requestResult(projects.get(projectId));
    if (project) projects.put({ ...project, updatedAt: now });
    await done;
    return pageRecord;
  } finally {
    database.close();
  }
}

export async function deleteWorkspacePage(projectId, pageNumber) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction([PAGE_STORE, PROJECT_STORE], "readwrite");
    const done = transactionDone(transaction);
    transaction.objectStore(PAGE_STORE).delete(`${projectId}:${pageNumber}`);

    const projects = transaction.objectStore(PROJECT_STORE);
    const project = await requestResult(projects.get(projectId));
    if (project) projects.put({ ...project, updatedAt: new Date().toISOString() });
    await done;
  } finally {
    database.close();
  }
}

export function projectFile(project) {
  if (!project?.pdfBlob) return null;
  return new File([project.pdfBlob], project.fileName || project.name || "textbook.pdf", {
    type: project.fileType || "application/pdf",
    lastModified: project.fileLastModified || Date.now(),
  });
}
