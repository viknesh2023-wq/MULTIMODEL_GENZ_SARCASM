import { motion } from 'framer-motion';

interface ProbabilityBarsProps {
  nonSarcastic: number;  // 0-1
  sarcastic: number;     // 0-1
}

export function ProbabilityBars({ nonSarcastic, sarcastic }: ProbabilityBarsProps) {
  return (
    <div className="space-y-4">
      <h4 className="text-xs text-zinc-400 uppercase tracking-wider font-medium">Probability Distribution</h4>

      {/* Non-Sarcastic */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="text-zinc-300 font-medium">Non-Sarcastic</span>
          </div>
          <span className="font-mono text-emerald-400 font-semibold">
            {(nonSarcastic * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2.5 rounded-full bg-zinc-800 overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-emerald-600"
            initial={{ width: 0 }}
            animate={{ width: `${nonSarcastic * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
          />
        </div>
      </div>

      {/* Sarcastic */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span className="text-zinc-300 font-medium">Sarcastic</span>
          </div>
          <span className="font-mono text-amber-400 font-semibold">
            {(sarcastic * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2.5 rounded-full bg-zinc-800 overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-amber-600"
            initial={{ width: 0 }}
            animate={{ width: `${sarcastic * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
          />
        </div>
      </div>
    </div>
  );
}
