import { cn } from '../../lib/utils'

export function Skeleton({ className, ...props }) {
  return (
    <div
      className={cn(
        'rounded-lg bg-surface-700/50 animate-pulse',
        className
      )}
      {...props}
    />
  )
}

export function SkeletonCard({ lines = 3 }) {
  return (
    <div className="rounded-2xl border border-surface-700/50 bg-surface-800/50 p-5 space-y-4">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-8 w-1/2" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-3 w-full" />
      ))}
    </div>
  )
}
