import { useEffect, useState } from 'react'
import { cn, getHealthGradient, getHealthTextColor } from '../../lib/utils'

export function HealthGauge({ score = 0, size = 180, label = 'Health Score' }) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100)
    return () => clearTimeout(timer)
  }, [score])

  const radius = (size - 24) / 2
  const circumference = Math.PI * radius
  const progress = (animatedScore / 100) * circumference
  const rotation = -180

  const getColor = (s) => {
    if (s >= 80) return { stroke: '#34d399', glow: 'rgba(52, 211, 153, 0.3)' }
    if (s >= 50) return { stroke: '#fbbf24', glow: 'rgba(251, 191, 36, 0.3)' }
    return { stroke: '#f87171', glow: 'rgba(248, 113, 113, 0.3)' }
  }

  const colors = getColor(animatedScore)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size / 2 + 20 }}>
        <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
          {/* Background arc */}
          <path
            d={`M ${12} ${size / 2 + 8} A ${radius} ${radius} 0 0 1 ${size - 12} ${size / 2 + 8}`}
            fill="none"
            stroke="rgba(30, 41, 59, 0.8)"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* Progress arc */}
          <path
            d={`M ${12} ${size / 2 + 8} A ${radius} ${radius} 0 0 1 ${size - 12} ${size / 2 + 8}`}
            fill="none"
            stroke={colors.stroke}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${progress} ${circumference}`}
            style={{
              transition: 'stroke-dasharray 1.5s cubic-bezier(0.4, 0, 0.2, 1)',
              filter: `drop-shadow(0 0 8px ${colors.glow})`,
            }}
          />
        </svg>
        {/* Center label */}
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
          <span className={cn('text-4xl font-bold tabular-nums', getHealthTextColor(animatedScore))}>
            {Math.round(animatedScore)}
          </span>
        </div>
      </div>
      <span className="metric-label">{label}</span>
    </div>
  )
}
