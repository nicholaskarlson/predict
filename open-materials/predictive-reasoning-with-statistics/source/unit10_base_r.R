#!/usr/bin/env Rscript
# Independently written base-R prediction, rank-AUC, confusion matrices and
# conditional student bootstrap. CSV inputs are frozen equivalents of JSON.
options(stringsAsFactors=FALSE,digits=17)
read_csv <- function(p) read.csv(p,check.names=FALSE,colClasses="character",na.strings=NULL)
num <- function(x) as.numeric(x)
yes <- function(x) as.integer(x)
member <- function(rows,g) {
 if(g$field=="all") return(rep(TRUE,nrow(rows)))
 if(g$field=="part_time_first_gen") return(paste(rows$part_time,rows$first_gen,sep="|")==g$value)
 rows[[g$field]]==g$value
}
forecast <- function(rows,b,transforms) {
 z<-function(field) { t<-transforms[transforms$feature==field,];if(nrow(t)!=1||num(t$sd)<=0)stop("frozen scale");(num(rows[[field]])-num(t$mean))/num(t$sd) }
 X<-cbind(intercept=1,part_time=num(rows$part_time),first_gen=num(rows$first_gen),z_age=z("age"),gender_male=yes(rows$gender=="male"),gender_another=yes(rows$gender=="another"),program_Social_Sciences=yes(rows$program_area=="Social Sciences"),program_STEM=yes(rows$program_area=="STEM"),program_Health=yes(rows$program_area=="Health"),program_Business=yes(rows$program_area=="Business"),z_prior_grade=z("prior_grade"),z_wellbeing_score=z("wellbeing_score"),z_stress_score=z("stress_score"),offer=0)
 if(!identical(colnames(X),b$term))stop("feature order")
 plogis(drop(X%*%num(b$coefficient)))
}
auc_score<-function(y,p) {
 a<-sum(y);b<-length(y)-a
 if(!a||!b)return(list(value=NA_real_,pairs=0,den=0,reason="one_class"))
 positive_ranks<-sum(rank(p,ties.method="average")[y==1]);pairs<-2*(positive_ranks-a*(a+1)/2);den<-2*a*b
 list(value=pairs/den,pairs=pairs,den=den,reason="")
}
prob_metrics<-function(y,p) {
 if(any(!is.finite(p)|p<0|p>1))stop("invalid probability")
 n<-length(y);events<-sum(y)
 if(!n)return(list(n=0,events=0,nonevents=0,auc=NA_real_,auc_pairs2=0,auc_denominator=0,auc_reason="empty_group",brier=NA_real_,log_loss=NA_real_,mean_probability=NA_real_,observed_fraction=NA_real_,calibration_bias=NA_real_,ece=NA_real_,clipped_n=0))
 auc<-auc_score(y,p);cl<-pmin(pmax(p,1e-12),1-1e-12);bin<-pmin(floor(p*5),4);ece<-sum(vapply(0:4,function(j){i<-bin==j;if(any(i))sum(i)/n*abs(mean(p[i])-mean(y[i])) else 0},0.))
 list(n=n,events=events,nonevents=n-events,auc=auc$value,auc_pairs2=auc$pairs,auc_denominator=auc$den,auc_reason=auc$reason,brier=mean((p-y)^2),log_loss=-mean(ifelse(y==1,log(cl),log1p(-cl))),mean_probability=mean(p),observed_fraction=mean(y),calibration_bias=mean(p)-mean(y),ece=ece,clipped_n=sum(p!=cl))
}
decisions<-function(rows,p,setting) {
 ranked<-order(-p,rows$row_id,seq_along(p));a<-integer(length(p));a[head(ranked,yes(setting$capacity))]<-1
 list(threshold_020=yes(p>=num(setting$threshold)),capacity=a)
}
action_metrics<-function(y,a) {
 n<-length(y);tab<-table(factor(y,levels=c(0,1)),factor(a,levels=c(0,1)));tn<-unname(tab[1,1]);fp<-unname(tab[1,2]);fn<-unname(tab[2,1]);tp<-unname(tab[2,2]);selected<-tp+fp;events<-tp+fn;nonevents<-fp+tn
 list(n=n,selected=selected,tp=tp,fp=fp,fn=fn,tn=tn,selection_rate=if(n)selected/n else NA_real_,tpr=if(events)tp/events else NA_real_,tpr_reason=if(events)"" else "no_events",fpr=if(nonevents)fp/nonevents else NA_real_,fpr_reason=if(nonevents)"" else "no_nonevents",ppv=if(selected)tp/selected else NA_real_,ppv_reason=if(selected)"" else "no_selected",net_benefit=if(n)(tp-.25*fp)/n else NA_real_)
}
calculate<-function(root) {
 policy<-paste(readLines(file.path(root,"model-specs/unit10.json"),warn=FALSE),collapse="\n")
 block<-regmatches(policy,regexpr('"predictor_allowlist"[[:space:]]*:[[:space:]]*\\[[^]]*\\]',policy))
 allowed<-strsplit(gsub('[[:space:]"\\[\\]]','',sub('^[^:]*:','',block),perl=TRUE),',',fixed=TRUE)[[1]]
 if(!identical(allowed,c("part_time","first_gen","age","gender","program_area","prior_grade","wellbeing_score","stress_score")))stop("information boundary")
 for(flag in c("fit_or_update","group_threshold_selection","final_labels_for_selection"))if(!grepl(paste0('"',flag,'"[[:space:]]*:[[:space:]]*false'),policy))stop("selection boundary")
 data<-read_csv(file.path(root,"data/unit10_rows.csv"));roles<-read_csv(file.path(root,"splits/unit10_roles.csv"));groups<-read_csv(file.path(root,"model-specs/unit10_groups.csv"));settings<-read_csv(file.path(root,"model-specs/unit10_actions.csv"));b<-read_csv(file.path(root,"model-specs/unit10_coefficients.csv"));transforms<-read_csv(file.path(root,"model-specs/unit10_transforms.csv"));draws<-read_csv(file.path(root,"splits/unit10_bootstrap_rows.csv"))
 if(any(!is.finite(num(settings$threshold)))||any(num(settings$threshold)<0|num(settings$threshold)>1)||any(!is.finite(num(settings$capacity)))||any(num(settings$capacity)<0|num(settings$capacity)!=yes(settings$capacity))||any(yes(settings$draws)<1)||anyDuplicated(settings$role))stop("invalid action or resampling specification")
 if(anyDuplicated(data$row_id)||anyDuplicated(roles$row_id)||!setequal(data$row_id,roles$row_id))stop("duplicate or unmatched ID")
 roles<-roles[match(data$row_id,roles$row_id),];if(any(data$college_id!=roles$college_id)||any(!roles$assignment%in%c("continuing_final","new_final","new_updating"))||any(data$offer!="0"))stop("role or policy")
 if(any(!data$binary_y%in%c("0","1",""))||any(!data$part_time%in%c("0","1"))||any(!data$first_gen%in%c("0","1"))||any(!data$gender%in%c("female","male","another"))||any(!data$program_area%in%c("Arts and Humanities","Social Sciences","STEM","Health","Business")))stop("label or group")
 for(k in c("age","prior_grade","wellbeing_score","stress_score"))if(any(data[[k]]!="" & !is.finite(num(data[[k]]))))stop("nonfinite predictor")
 result<-setNames(vector("list",8),c("predictions","actions","scores","action_scores","bins","bootstrap","bootstrap_summary","toy"));add<-function(key,row){result[[key]][[length(result[[key]])+1]]<<-row}
 for(role in settings$role) {
  setting<-settings[settings$role==role,];rows<-data[roles$assignment==role & data$binary_y!="" & data$wellbeing_score!="" & data$stress_score!="",];p<-forecast(rows,b,transforms);y<-yes(rows$binary_y);rules<-decisions(rows,p,setting)
  if(yes(setting$capacity)>nrow(rows))stop("capacity exceeds role")
  for(i in seq_len(nrow(rows))) {
   add("predictions",list(role=role,row_id=rows$row_id[i],college_id=rows$college_id[i],y=y[i],probability=p[i]))
   for(rule in names(rules))add("actions",list(role=role,row_id=rows$row_id[i],rule=rule,selected=rules[[rule]][i]))
  }
  for(j in seq_len(nrow(groups))) {
   g<-groups[j,];ix<-member(rows,g);add("scores",c(list(role=role,group=g$id),prob_metrics(y[ix],p[ix])))
   for(rule in names(rules))add("action_scores",c(list(role=role,group=g$id,rule=rule),action_metrics(y[ix],rules[[rule]][ix])))
   for(bin in 0:4) {
    sel<-ix & pmin(floor(p*5),4)==bin;n<-sum(sel);add("bins",list(role=role,group=g$id,bin=bin,n=n,events=sum(y[sel]),mean_probability=if(n)mean(p[sel]) else NA_real_,observed_fraction=if(n)mean(y[sel]) else NA_real_,reason=if(n)"" else "empty_bin"))
   }
  }
  local<-draws[draws$role==role,];samples<-split(local,yes(local$draw_id))
  if(!identical(names(samples),as.character(seq_len(yes(setting$draws)))))stop("draw count")
  for(d in seq_len(yes(setting$draws))) {
   draw<-samples[[as.character(d)]];take<-match(draw$row_id,rows$row_id)
   if(anyNA(take)||any(rows$college_id[take]!=draw$college_id)||nrow(draw)!=nrow(rows))stop("resample role or college")
   for(college in unique(rows$college_id)) {pos<-yes(draw$position[draw$college_id==college]);n<-sum(rows$college_id==college);if(!identical(sort(pos),seq_len(n)))stop("resample positions")}
   sample<-rows[take,];sp<-p[take];sy<-y[take];srules<-decisions(sample,sp,setting)
   for(j in seq_len(nrow(groups))) {
    g<-groups[j,];ix<-member(sample,g);n<-sum(ix);add("bootstrap",list(role=role,draw_id=d,group=g$id,rule="probability",n=n,brier=if(n)mean((sp[ix]-sy[ix])^2) else NA_real_,calibration_bias=if(n)mean(sp[ix])-mean(sy[ix]) else NA_real_,selection_rate=NA_real_,tpr=NA_real_,fpr=NA_real_,ppv=NA_real_))
    for(rule in names(srules)){z<-action_metrics(sy[ix],srules[[rule]][ix]);add("bootstrap",c(list(role=role,draw_id=d,group=g$id,rule=rule,n=n,brier=NA_real_,calibration_bias=NA_real_),z[c("selection_rate","tpr","fpr","ppv")]))}
   }
  }
 }
 boot<-do.call(rbind,lapply(result$bootstrap,function(z)as.data.frame(z,stringsAsFactors=FALSE)))
 for(role in settings$role)for(j in seq_len(nrow(groups)))for(rule in c("probability","threshold_020","capacity")) {
  group<-groups$id[j];subset<-boot[boot$role==role & boot$group==group & boot$rule==rule,];metrics<-if(rule=="probability")c("brier","calibration_bias") else c("selection_rate","tpr","fpr","ppv")
  for(metric in metrics) {v<-subset[[metric]];v<-v[!is.na(v)];q<-if(length(v))as.numeric(quantile(v,c(.025,.5,.975),type=7)) else rep(NA_real_,3);add("bootstrap_summary",list(role=role,group=group,rule=rule,metric=metric,draws=nrow(subset),defined_draws=length(v),undefined_draws=nrow(subset)-length(v),lower=q[1],median=q[2],upper=q[3],reason=if(length(v))"" else "no_defined_replicates"))}
 }
 for(i in 1:2){ev<-c(10,40)[i];ne<-100-ev;y<-c(rep(1,ev),rep(0,ne));a<-c(rep(1,.8*ev),rep(0,.2*ev),rep(1,.1*ne),rep(0,.9*ne));add("toy",c(list(group=c("A","B")[i],prevalence=ev/100),action_metrics(y,a)))}
 result
}
emit<-function(result,out) {
 dir.create(out,recursive=TRUE,showWarnings=FALSE)
 for(key in names(result)) {
  rows<-lapply(result[[key]],function(z){for(k in names(z)){if(is.numeric(z[[k]]))z[[k]]<-if(is.na(z[[k]]))"" else sprintf("%.17g",z[[k]])};as.data.frame(z,stringsAsFactors=FALSE)})
  write.table(do.call(rbind,rows),file.path(out,paste0(key,".csv")),sep=",",row.names=FALSE,col.names=TRUE,quote=FALSE,na="")
 }
}
if(sys.nframe()==0){args<-commandArgs(trailingOnly=TRUE);if(length(args)!=2)stop("resource and output paths required");emit(calculate(args[1]),args[2]);cat("UNIT10_BASE_R_OK\n")}
