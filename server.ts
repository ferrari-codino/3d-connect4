import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "10mb" }));

// -------------------------------------------------------------
// Data Directories & File Paths
// -------------------------------------------------------------
const DATA_DIR = path.join(process.cwd(), "data");
if (!fs.existsSync(DATA_DIR)) {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  } catch (e) {
    console.error("Failed to create data directory:", e);
  }
}

const MODEL_FILE = path.join(DATA_DIR, "ai_learned_model.json");
const GAMES_FILE = path.join(DATA_DIR, "winning_games.json");

// -------------------------------------------------------------
// 3D Four-in-a-Row Core Game & AI Engine for Server Simulation
// -------------------------------------------------------------
const WINNING_LINES: number[][] = [];
const CELL_LINE_COUNTS = new Uint8Array(64);

function initWinningLines() {
  WINNING_LINES.length = 0;
  CELL_LINE_COUNTS.fill(0);
  const getIdx = (x: number, y: number, z: number) => x + y * 4 + z * 16;

  // 1. X direction (16 lines)
  for (let y = 0; y < 4; y++) {
    for (let z = 0; z < 4; z++) {
      WINNING_LINES.push([getIdx(0, y, z), getIdx(1, y, z), getIdx(2, y, z), getIdx(3, y, z)]);
    }
  }
  // 2. Y direction (16 lines)
  for (let x = 0; x < 4; x++) {
    for (let z = 0; z < 4; z++) {
      WINNING_LINES.push([getIdx(x, 0, z), getIdx(x, 1, z), getIdx(x, 2, z), getIdx(x, 3, z)]);
    }
  }
  // 3. Z direction (16 lines)
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      WINNING_LINES.push([getIdx(x, y, 0), getIdx(x, y, 1), getIdx(x, y, 2), getIdx(x, y, 3)]);
    }
  }
  // 4. XY plane diagonals (8 lines)
  for (let z = 0; z < 4; z++) {
    WINNING_LINES.push([getIdx(0, 0, z), getIdx(1, 1, z), getIdx(2, 2, z), getIdx(3, 3, z)]);
    WINNING_LINES.push([getIdx(3, 0, z), getIdx(2, 1, z), getIdx(1, 2, z), getIdx(0, 3, z)]);
  }
  // 5. XZ plane diagonals (8 lines)
  for (let y = 0; y < 4; y++) {
    WINNING_LINES.push([getIdx(0, y, 0), getIdx(1, y, 1), getIdx(2, y, 2), getIdx(3, y, 3)]);
    WINNING_LINES.push([getIdx(3, y, 0), getIdx(2, y, 1), getIdx(1, y, 2), getIdx(0, y, 3)]);
  }
  // 6. YZ plane diagonals (8 lines)
  for (let x = 0; x < 4; x++) {
    WINNING_LINES.push([getIdx(x, 0, 0), getIdx(x, 1, 1), getIdx(x, 2, 2), getIdx(x, 3, 3)]);
    WINNING_LINES.push([getIdx(x, 3, 0), getIdx(x, 2, 1), getIdx(x, 1, 2), getIdx(x, 0, 3)]);
  }
  // 7. Space 3D diagonals (4 lines)
  WINNING_LINES.push([getIdx(0, 0, 0), getIdx(1, 1, 1), getIdx(2, 2, 2), getIdx(3, 3, 3)]);
  WINNING_LINES.push([getIdx(3, 0, 0), getIdx(2, 1, 1), getIdx(1, 2, 2), getIdx(0, 3, 3)]);
  WINNING_LINES.push([getIdx(0, 3, 0), getIdx(1, 2, 1), getIdx(2, 1, 2), getIdx(3, 0, 3)]);
  WINNING_LINES.push([getIdx(3, 3, 0), getIdx(2, 2, 1), getIdx(1, 1, 2), getIdx(0, 0, 3)]);

  for (let i = 0; i < WINNING_LINES.length; i++) {
    const line = WINNING_LINES[i];
    for (let j = 0; j < 4; j++) {
      CELL_LINE_COUNTS[line[j]]++;
    }
  }
}
initWinningLines();

interface Move {
  player: number;
  col: number;
}

interface WinningGame {
  id: string;
  nickname: string;
  difficulty: string;
  order: string;
  totalMoves: number;
  moves: Move[];
  winningLine: number[];
  finalCol: number;
  dateStr: string;
  timestamp: number;
}

interface AiModelData {
  learnedBook: Record<string, number>;
  posBonuses: Record<string, number>;
  meta: {
    lastTrainedAt: string;
    totalGamesTrained: number;
    learnedCount: number;
    totalSimulatedGames: number;
    rating: number;
  };
}

let globalAiModel: AiModelData = {
  learnedBook: {},
  posBonuses: {},
  meta: {
    lastTrainedAt: new Date().toLocaleString("ja-JP"),
    totalGamesTrained: 0,
    learnedCount: 0,
    totalSimulatedGames: 0,
    rating: 2000,
  },
};

let storedGames: WinningGame[] = [];

function loadServerStorage() {
  try {
    if (fs.existsSync(MODEL_FILE)) {
      const raw = fs.readFileSync(MODEL_FILE, "utf-8");
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        globalAiModel = parsed;
      }
    }
    if (fs.existsSync(GAMES_FILE)) {
      const raw = fs.readFileSync(GAMES_FILE, "utf-8");
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        storedGames = parsed;
      }
    }
  } catch (e) {
    console.warn("Error loading server persistent storage:", e);
  }
}
loadServerStorage();

function saveServerStorage() {
  try {
    fs.writeFileSync(MODEL_FILE, JSON.stringify(globalAiModel, null, 2), "utf-8");
    fs.writeFileSync(GAMES_FILE, JSON.stringify(storedGames, null, 2), "utf-8");
  } catch (e) {
    console.error("Error saving server persistent storage:", e);
  }
}

// -------------------------------------------------------------
// AI Rating Calculator
// -------------------------------------------------------------
function calculateRating(totalGames: number, learnedCount: number, posBonuses: Record<string, number>, simulatedGames: number): number {
  const BASE_RATING = 2000;
  let posSum = 0;
  if (posBonuses) {
    Object.values(posBonuses).forEach((v) => {
      if (typeof v === "number") posSum += v;
    });
  }
  const humanGain = totalGames * 14 + learnedCount * 3 + Math.min(300, Math.floor(posSum / 8));
  const simGain = Math.floor(Math.sqrt(simulatedGames) * 4.2);
  return BASE_RATING + humanGain + simGain;
}

// -------------------------------------------------------------
// Training Logic
// -------------------------------------------------------------
function trainServerAiModel(): AiModelData {
  const learnedBook: Record<string, number> = { ...globalAiModel.learnedBook };
  const posBonuses: Record<string, number> = { ...globalAiModel.posBonuses };
  let learnedCount = Object.keys(learnedBook).length;

  storedGames.forEach((game) => {
    if (!game.moves || !Array.isArray(game.moves)) return;

    const simBoard = new Array(64).fill(0);
    const simHeights = new Array(16).fill(0);

    for (let step = 0; step < game.moves.length && step < 8; step++) {
      const m = game.moves[step];
      const boardKey = simBoard.join("");

      if (m.player === (game.order === "先手" ? 1 : 2)) {
        if (learnedBook[boardKey] === undefined) {
          learnedBook[boardKey] = m.col;
          learnedCount++;
        }
      }

      const z = simHeights[m.col];
      if (z < 4) {
        simBoard[m.col + z * 16] = m.player;
        simHeights[m.col]++;
      }
    }

    if (game.winningLine && Array.isArray(game.winningLine)) {
      game.winningLine.forEach((idx) => {
        posBonuses[idx] = (posBonuses[idx] || 0) + 12;
      });
    }

    if (game.finalCol >= 0 && game.finalCol < 16) {
      for (let z = 0; z < 4; z++) {
        const cIdx = game.finalCol + z * 16;
        posBonuses[cIdx] = (posBonuses[cIdx] || 0) + 8;
      }
    }
  });

  const rating = calculateRating(
    storedGames.length,
    learnedCount,
    posBonuses,
    globalAiModel.meta.totalSimulatedGames || 0
  );

  globalAiModel = {
    learnedBook,
    posBonuses,
    meta: {
      lastTrainedAt: new Date().toLocaleString("ja-JP"),
      totalGamesTrained: storedGames.length,
      learnedCount,
      totalSimulatedGames: globalAiModel.meta.totalSimulatedGames || 0,
      rating,
    },
  };

  saveServerStorage();
  return globalAiModel;
}

// -------------------------------------------------------------
// High-Speed Self-Play Engine (Pure Memory & Mathematical Sim)
// -------------------------------------------------------------
function checkBoardWin(b: Uint8Array): { winner: number; line: number[] } | null {
  for (let i = 0; i < 76; i++) {
    const line = WINNING_LINES[i];
    const val = b[line[0]];
    if (val !== 0 && val === b[line[1]] && val === b[line[2]] && val === b[line[3]]) {
      return { winner: val, line };
    }
  }
  return null;
}

function evaluateSimBoard(b: Uint8Array, h: Uint8Array, evalPlayer: number, bonuses: Record<string, number>): number {
  const oppPlayer = evalPlayer === 1 ? 2 : 1;
  let score = 0;
  let pImmediate = 0, oImmediate = 0;

  for (let i = 0; i < 76; i++) {
    const line = WINNING_LINES[i];
    let pCount = 0, oCount = 0, emptyIdx = -1;

    for (let j = 0; j < 4; j++) {
      const val = b[line[j]];
      if (val === evalPlayer) pCount++;
      else if (val === oppPlayer) oCount++;
      else emptyIdx = line[j];
    }

    if (pCount > 0 && oCount > 0) continue;

    if (pCount === 3) {
      const col = emptyIdx % 16;
      const z = Math.floor(emptyIdx / 16);
      if (h[col] === z) {
        pImmediate++;
        score += 850;
      } else {
        score += 260;
      }
    } else if (pCount === 2) {
      score += 40;
    }

    if (oCount === 3) {
      const col = emptyIdx % 16;
      const z = Math.floor(emptyIdx / 16);
      if (h[col] === z) {
        oImmediate++;
        score -= 900;
      } else {
        score -= 300;
      }
    } else if (oCount === 2) {
      score -= 60;
    }
  }

  if (pImmediate >= 2) return 8800;
  if (oImmediate >= 2) return -8800;

  // Center / Key square weights
  for (let idx = 0; idx < 64; idx++) {
    const val = b[idx];
    if (val === 0) continue;
    const baseVal = CELL_LINE_COUNTS[idx] * 2 + (bonuses[idx] || 0);
    if (val === evalPlayer) score += baseVal;
    else score -= baseVal;
  }

  return score;
}

function chooseFastSimMove(
  b: Uint8Array,
  h: Uint8Array,
  player: number,
  depth: number,
  book: Record<string, number>,
  bonuses: Record<string, number>,
  epsilon: number = 0.05
): number {
  // Check book
  const key = b.join("");
  if (book[key] !== undefined && h[book[key]] < 4) {
    return book[key];
  }

  // Valid moves
  const validCols: number[] = [];
  for (let c = 0; c < 16; c++) {
    if (h[c] < 4) validCols.push(c);
  }
  if (validCols.length === 0) return -1;
  if (validCols.length === 1) return validCols[0];

  // 1. Immediate Win
  for (let col of validCols) {
    const z = h[col];
    const idx = col + z * 16;
    b[idx] = player;
    const win = checkBoardWin(b);
    b[idx] = 0;
    if (win && win.winner === player) return col;
  }

  // 2. Immediate Block
  const opp = player === 1 ? 2 : 1;
  for (let col of validCols) {
    const z = h[col];
    const idx = col + z * 16;
    b[idx] = opp;
    const win = checkBoardWin(b);
    b[idx] = 0;
    if (win && win.winner === opp) return col;
  }

  // Exploration epsilon (random strategic move among top 3)
  if (Math.random() < epsilon) {
    const centers = [5, 6, 9, 10, 0, 3, 12, 15].filter((c) => h[c] < 4);
    if (centers.length > 0) {
      return centers[Math.floor(Math.random() * centers.length)];
    }
  }

  // Alpha-beta search (Depth 2~3)
  let bestScore = -Infinity;
  let bestCol = validCols[0];

  for (let col of validCols) {
    const z = h[col];
    const idx = col + z * 16;
    b[idx] = player;
    h[col]++;

    let score = -evaluateSimBoard(b, h, opp, bonuses);

    // One level deeper lookahead for opponent trap avoidance
    if (depth >= 2) {
      let oppBest = -Infinity;
      for (let c2 = 0; c2 < 16; c2++) {
        if (h[c2] < 4) {
          const z2 = h[c2];
          const idx2 = c2 + z2 * 16;
          b[idx2] = opp;
          h[c2]++;
          const win2 = checkBoardWin(b);
          let s2 = win2 ? 9000 : evaluateSimBoard(b, h, opp, bonuses);
          b[idx2] = 0;
          h[c2]--;
          if (s2 > oppBest) oppBest = s2;
        }
      }
      score -= (oppBest > -Infinity ? oppBest * 0.5 : 0);
    }

    b[idx] = 0;
    h[col]--;

    if (score > bestScore) {
      bestScore = score;
      bestCol = col;
    }
  }

  return bestCol;
}

// -------------------------------------------------------------
// Background Simulation Manager
// -------------------------------------------------------------
interface SimStatus {
  isRunning: boolean;
  targetCount: number;
  currentCount: number;
  p1Wins: number;
  p2Wins: number;
  draws: number;
  learnedMovesCount: number;
  startedAt: string | null;
  speedGamesPerSec: number;
  progressPct: number;
  etaSec: number;
}

let simStatus: SimStatus = {
  isRunning: false,
  targetCount: 0,
  currentCount: 0,
  p1Wins: 0,
  p2Wins: 0,
  draws: 0,
  learnedMovesCount: 0,
  startedAt: null,
  speedGamesPerSec: 0,
  progressPct: 0,
  etaSec: 0,
};

let simCancelRequested = false;

async function runServerSelfPlayBatch(targetGames: number) {
  if (simStatus.isRunning) return;

  simStatus = {
    isRunning: true,
    targetCount: targetGames,
    currentCount: 0,
    p1Wins: 0,
    p2Wins: 0,
    draws: 0,
    learnedMovesCount: 0,
    startedAt: new Date().toLocaleString("ja-JP"),
    speedGamesPerSec: 0,
    progressPct: 0,
    etaSec: 0,
  };
  simCancelRequested = false;

  const startTime = Date.now();
  let lastCalcTime = startTime;
  let lastCount = 0;

  const localBook = { ...globalAiModel.learnedBook };
  const localBonuses = { ...globalAiModel.posBonuses };
  let newLearned = 0;

  const board = new Uint8Array(64);
  const heights = new Uint8Array(16);

  for (let g = 0; g < targetGames; g++) {
    if (simCancelRequested) break;

    board.fill(0);
    heights.fill(0);
    const moves: Move[] = [];
    let currentPlayer = 1;
    let winner: number | null = null;
    let winLine: number[] | null = null;
    let finalCol = -1;

    for (let turn = 0; turn < 64; turn++) {
      const col = chooseFastSimMove(
        board,
        heights,
        currentPlayer,
        2,
        localBook,
        localBonuses,
        0.08
      );
      if (col === -1) break;

      const z = heights[col];
      const idx = col + z * 16;
      board[idx] = currentPlayer;
      heights[col]++;
      moves.push({ player: currentPlayer, col });

      const win = checkBoardWin(board);
      if (win) {
        winner = win.winner;
        winLine = win.line;
        finalCol = col;
        break;
      }

      currentPlayer = currentPlayer === 1 ? 2 : 1;
    }

    if (winner === 1) simStatus.p1Wins++;
    else if (winner === 2) simStatus.p2Wins++;
    else simStatus.draws++;

    // Reinforcement Learning: Extract winning patterns from victorious play
    if (winner !== null && winLine !== null && moves.length >= 4) {
      const simB = new Uint8Array(64);
      const simH = new Uint8Array(16);

      for (let s = 0; s < moves.length && s < 8; s++) {
        const m = moves[s];
        const key = simB.join("");
        if (m.player === winner && localBook[key] === undefined) {
          localBook[key] = m.col;
          newLearned++;
        }
        const z = simH[m.col];
        if (z < 4) {
          simB[m.col + z * 16] = m.player;
          simH[m.col]++;
        }
      }

      winLine.forEach((cIdx) => {
        localBonuses[cIdx] = (localBonuses[cIdx] || 0) + 8;
      });
      if (finalCol >= 0 && finalCol < 16) {
        for (let z = 0; z < 4; z++) {
          const cIdx = finalCol + z * 16;
          localBonuses[cIdx] = (localBonuses[cIdx] || 0) + 4;
        }
      }
    }

    simStatus.currentCount = g + 1;
    simStatus.learnedMovesCount = newLearned;
    simStatus.progressPct = Math.round(((g + 1) / targetGames) * 100);

    const now = Date.now();
    if (now - lastCalcTime > 500) {
      const elapsedSec = (now - startTime) / 1000;
      const recentSpeed = (g + 1 - lastCount) / ((now - lastCalcTime) / 1000);
      simStatus.speedGamesPerSec = Math.round(recentSpeed);
      const remaining = targetGames - (g + 1);
      simStatus.etaSec = recentSpeed > 0 ? Math.round(remaining / recentSpeed) : 0;

      lastCalcTime = now;
      lastCount = g + 1;

      // Yield event loop
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Update Global AI Model with simulation learnings
  globalAiModel.learnedBook = localBook;
  globalAiModel.posBonuses = localBonuses;
  globalAiModel.meta.totalSimulatedGames = (globalAiModel.meta.totalSimulatedGames || 0) + simStatus.currentCount;
  globalAiModel.meta.learnedCount = Object.keys(localBook).length;
  globalAiModel.meta.lastTrainedAt = new Date().toLocaleString("ja-JP");
  globalAiModel.meta.rating = calculateRating(
    storedGames.length,
    globalAiModel.meta.learnedCount,
    globalAiModel.posBonuses,
    globalAiModel.meta.totalSimulatedGames
  );

  saveServerStorage();

  simStatus.isRunning = false;
  simStatus.progressPct = 100;
  simStatus.etaSec = 0;
  console.log(
    `[Self-Play Sim Completed] Games: ${simStatus.currentCount}, New Moves: ${newLearned}, Rating: ${globalAiModel.meta.rating}`
  );
}

// -------------------------------------------------------------
// Daily Automated Cron / Scheduler (Every 24h at 03:00)
// -------------------------------------------------------------
function startDailyAutoScheduler() {
  setInterval(() => {
    const now = new Date();
    // Run at 03:00 AM server time
    if (now.getHours() === 3 && now.getMinutes() === 0 && !simStatus.isRunning) {
      console.log("[Daily Auto AI Evolution] Starting daily 2000-game self-play & retraining...");
      trainServerAiModel();
      runServerSelfPlayBatch(2000);
    }
  }, 60 * 1000);
}
startDailyAutoScheduler();

// -------------------------------------------------------------
// API Routes
// -------------------------------------------------------------
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: Date.now() });
});

// 1. Get Latest Global AI Model
app.get("/api/ai/model", (req, res) => {
  res.json(globalAiModel);
});

// 2. Record Human Winning Game (When human defeats AI)
app.post("/api/games/record", (req, res) => {
  try {
    const game: WinningGame = req.body;
    if (!game || !game.moves || !Array.isArray(game.moves)) {
      return res.status(400).json({ error: "Invalid game payload" });
    }
    game.id = game.id || `game_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    game.timestamp = game.timestamp || Date.now();
    game.dateStr = game.dateStr || new Date().toLocaleString("ja-JP");

    storedGames.unshift(game);
    if (storedGames.length > 500) storedGames = storedGames.slice(0, 500);

    // Auto-retrain with the newly learned human victory
    const updatedModel = trainServerAiModel();

    res.json({
      success: true,
      gameId: game.id,
      totalGames: storedGames.length,
      model: updatedModel,
    });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// 3. List Recorded Defeat Games
app.get("/api/games/list", (req, res) => {
  res.json({
    total: storedGames.length,
    games: storedGames.slice(0, 100),
  });
});

// 4. Trigger Server-Side Self-Play Simulation (Plan B)
app.post("/api/simulation/start", async (req, res) => {
  const count = parseInt(req.body.count || "1000", 10);
  const clampedCount = Math.max(100, Math.min(100000, isNaN(count) ? 1000 : count));

  if (simStatus.isRunning) {
    return res.status(409).json({ error: "Simulation is already running", status: simStatus });
  }

  // Start asynchronously in background
  runServerSelfPlayBatch(clampedCount);

  res.json({
    success: true,
    message: `Started server self-play simulation with ${clampedCount} games.`,
    status: simStatus,
  });
});

// 5. Get Simulation Status
app.get("/api/simulation/status", (req, res) => {
  res.json({
    status: simStatus,
    modelRating: globalAiModel.meta.rating,
  });
});

// 6. Stop Simulation
app.post("/api/simulation/stop", (req, res) => {
  simCancelRequested = true;
  res.json({ success: true, message: "Simulation cancel requested." });
});

// 7. Reset AI Learning Data
app.post("/api/ai/reset", (req, res) => {
  globalAiModel = {
    learnedBook: {},
    posBonuses: {},
    meta: {
      lastTrainedAt: "初期化済",
      totalGamesTrained: 0,
      learnedCount: 0,
      totalSimulatedGames: 0,
      rating: 2000,
    },
  };
  storedGames = [];
  saveServerStorage();
  res.json({ success: true, model: globalAiModel });
});

// -------------------------------------------------------------
// Vite Middleware / Static Serving
// -------------------------------------------------------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`🚀 3D Connect Four server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
