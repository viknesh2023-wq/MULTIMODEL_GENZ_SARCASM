import { motion } from 'framer-motion';

interface ConfusionMatrixProps {
  matrix: number[][];
  labels?: string[];
}

export function ConfusionMatrix({ matrix, labels = ['Non-Sarcastic', 'Sarcastic'] }: ConfusionMatrixProps) {
  const flat = matrix.flat();
  const total = flat.reduce((a, b) => a + b, 0);

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
      <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-6">
        Confusion Matrix Heatmap
      </h3>

      {/* Column header */}
      <div className="flex">
        <div className="w-32 shrink-0" />
        <div className="flex-1">
          <div className="text-center text-[11px] text-zinc-500 mb-2 tracking-wider font-semibold">
            PREDICTED
          </div>
          <div
            className="grid gap-2 mb-1"
            style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))` }}
          >
            {labels.map(l => (
              <div key={l} className="text-center text-xs text-zinc-300 font-medium">{l}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Rows */}
      <div className="flex items-center gap-2">
        <div className="w-4 shrink-0">
          <div
            className="text-[11px] text-zinc-500 tracking-wider font-semibold"
            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', whiteSpace: 'nowrap' }}
          >
            ACTUAL
          </div>
        </div>
        <div className="flex-1">
          {matrix.map((row, rowIdx) => (
            <div key={rowIdx} className="flex items-center gap-2 mb-2">
              <div className="w-28 text-xs text-zinc-300 font-medium text-right pr-2 shrink-0">
                {labels[rowIdx]}
              </div>
              <div
                className="grid gap-2 flex-1"
                style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))` }}
              >
                {row.map((value, colIdx) => {
                  const isCorrect = rowIdx === colIdx;
                  return (
                    <motion.div
                      key={colIdx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: (rowIdx * labels.length + colIdx) * 0.05 }}
                      className={`rounded-lg p-3 text-center border ${
                        isCorrect
                          ? 'bg-zinc-800/80 border-zinc-700 text-white'
                          : 'bg-zinc-950 border-zinc-800 text-zinc-400'
                      }`}
                    >
                      <div className="text-lg font-bold font-mono">
                        {value.toLocaleString('en-IN')}
                      </div>
                      <div className="text-xs text-zinc-500 mt-0.5">
                        {((value / total) * 100).toFixed(1)}%
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-4 justify-end text-xs text-zinc-500">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-zinc-700 border border-zinc-600" />
          Correct predictions (74,507)
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-zinc-950 border border-zinc-800" />
          Misclassifications (22,012)
        </div>
      </div>
    </div>
  );
}
