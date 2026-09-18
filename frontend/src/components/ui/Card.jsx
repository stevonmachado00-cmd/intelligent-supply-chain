import { cn } from '../../lib/utils'

export function Card({ className, children, hover = false, ...props }) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-surface-700/50 bg-surface-800/50 backdrop-blur-xl p-5',
        hover && 'transition-all duration-300 hover:border-brand-500/30 hover:shadow-glow cursor-pointer',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children, ...props }) {
  return (
    <div className={cn('mb-4', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className, children, ...props }) {
  return (
    <h3 className={cn('text-sm font-semibold text-white', className)} {...props}>
      {children}
    </h3>
  )
}

export function CardDescription({ className, children, ...props }) {
  return (
    <p className={cn('text-xs text-slate-400 mt-1', className)} {...props}>
      {children}
    </p>
  )
}
