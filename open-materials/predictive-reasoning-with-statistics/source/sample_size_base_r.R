# Independent base R: three-criterion binary model-development planning.
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2)
resource <- args[1];out <- args[2]
read <- function(name) read.csv(file.path(resource,name),stringsAsFactors=FALSE,check.names=FALSE)
plan <- function(p,pi,r2,s,delta,margin,z,fraction) {
  maximum <- 1-exp(2*(pi*log(pi)+(1-pi)*log(1-pi)))
  stopifnot(p>0,pi>0,pi<1,r2>0,r2<min(s,maximum),s>0,s<1,delta>0,margin>0,z>0,fraction>0,fraction<=1)
  s2 <- r2/(r2+delta*maximum)
  raw <- c(p/((s-1)*log(1-r2/s)),p/((s2-1)*log(1-r2/s2)),z^2*pi*(1-pi)/margin^2)
  n <- ceiling(raw); required <- max(n)
  data.frame(parameters=p,prevalence=pi,r2_cs=r2,r2_max=maximum,shrinkage=s,optimism_delta=delta,risk_margin=margin,z=z,usable_fraction=fraction,optimism_shrinkage=s2,n_shrinkage_raw=raw[1],n_optimism_raw=raw[2],n_risk_raw=raw[3],n_shrinkage=n[1],n_optimism=n[2],n_risk=n[3],n_required=required,expected_events=required*pi,invitations=ceiling(required/fraction))
}
diagnostics <- list();scenarios <- list()
for (unit in c(7,9)) {
  label <- if(unit==7) 'unit07_full' else 'unit09_refit'
  model <- if(unit==7) 'full_logistic' else 'update_refit'
  key <- if(unit==7) 'model_id' else 'fit_id'
  d <- read(sprintf('data/unit%02d_rows.csv',unit))
  if (unit==7) y <- d$y[d$role=='development'] else {
    roles <- read('splits/unit09_roles.csv')
    y <- d$binary_y[d$row_id %in% roles$row_id[roles$assignment=='new_updating']]
  }
  n <- length(y);events <- sum(y);pi <- mean(y)
  co <- read(sprintf('expected-results/unit%02d_coefficients.csv',unit));p <- sum(co[[key]]==model & co$term!='intercept')
  di <- read(sprintf('expected-results/unit%02d_diagnostics.csv',unit));di <- di[di[[key]]==model,]
  stopifnot(nrow(di)==1,di$n_fit==n,di$events==events,di$p==p+1)
  nll <- di$nll;null <- -(events*log(pi)+(n-events)*log(1-pi));lr <- 2*(null-nll)
  apparent <- 1-exp(-lr/n);maximum <- 1-exp(-2*null/n)
  diagnostics[[label]] <- data.frame(case=label,n=n,events=events,parameters=p,prevalence=pi,model_nll=nll,null_nll=null,likelihood_ratio=lr,heuristic_shrinkage=1-p/lr,r2_apparent=apparent,r2_max=maximum)
  for (name in c('apparent_plugin_optimistic','assumed_15pct_max')) {
    r2 <- if(name=='apparent_plugin_optimistic') apparent else .15*maximum
    scenarios[[paste0(label,name)]] <- cbind(data.frame(scenario=paste0(label,'_',name),origin='retrospective_assumption',available_n=n),plan(p,pi,r2,.9,.05,.05,1.96,1))
  }
}
# CSV mirrors the declared JSON assumptions; the gate requires exact agreement.
assumptions <- read('model-specs/sample_size_scenarios.csv')
for (i in seq_len(nrow(assumptions))) {
  a <- assumptions[i,]
  scenarios[[a$id]] <- cbind(data.frame(scenario=a$id,origin='prospective_assumption',available_n=0),plan(a$parameters,a$prevalence,a$r2_cs,a$shrinkage,a$optimism_delta,a$risk_margin,a$z,a$usable_fraction))
}
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(out,'planning_diagnostics.csv'),row.names=FALSE)
write.csv(do.call(rbind,scenarios),file.path(out,'planning_scenarios.csv'),row.names=FALSE)
