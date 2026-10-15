import { motion } from 'framer-motion';

interface ConfidenceGaugeProps {
  confidence: number;   // 0-1
  label: string;        // e.g. "SARCASTIC"
  isSarcastic: boolean;
}

export function ConfidenceGauge({ confidence, label, isSarcastic }: ConfidenceGaugeProps) {
  const radius = 80;
  const stroke = 8;
  const normalizedRadius = radius - stroke / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const arcLength = circumference * 0.75;
  const fillLength = arcLength * confidence;
  const gapLength = circumference - arcLength;

  // Fixed understated colors
  const color = isSarcastic ? '#d97706' : '#059669';

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative" style={{ width: radius * 2, height: radius * 2 }}>
        <svg
          width={radius * 2}
          height={radius * 2}
          viewBox={`0 0 ${radius * 2} ${radius * 2}`}
          style={{ transform: 'rotate(135deg)' }}
        >
          {/* Track */}
          <circle
            cx={radius} cy={radius} r={normalizedRadius}
            fill="none" stroke="#27272a"
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${gapLength}`}
            strokeLinecap="round"
          />
          {/* Animated fill */}
          <motion.circle
            cx={radius} cy={radius} r={normalizedRadius}
            fill="none" stroke={color}
            strokeWidth={stroke}
            strokeDasharray={`${fillLength} ${circumference - fillLength}`}
            strokeLinecap="round"
            initial={{ strokeDasharray: `0 ${circumference}` }}
            animate={{ strokeDasharray: `${fillLength} ${circumference - fillLength}` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.1 }}
          />
        </svg>
        {/* Centre text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div
            className="text-3xl font-bold font-mono text-white"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            {(confidence * 100).toFixed(1)}%
          </motion.div>
          <div className="text-[11px] text-zinc-400 mt-1 uppercase tracking-wider font-medium">Confidence</div>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        className="text-center"
      >
        <div
          className={`text-xl font-bold tracking-wider px-6 py-2 rounded-lg border ${
            isSarcastic
              ? 'bg-amber-950/40 border-amber-800/80 text-amber-200'
              : 'bg-emerald-950/40 border-emerald-800/80 text-emerald-200'
          }`}
        >
          {label}
        </div>
      </motion.div>
    </div>
  );
}
