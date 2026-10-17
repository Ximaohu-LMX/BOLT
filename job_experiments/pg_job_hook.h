/* JOB predicate-keyed replacement at the inherited restriction hook.
 * Never changes joins; all four modeled columns must map explicitly.
 * Files are local to a dedicated experiment cluster, not a public service.
 */
#include <math.h>
#include "lib/stringinfo.h"
#include "utils/json.h"
#define JOB_MAX_KEYS 256
#define JOB_KEY_BYTES 16384
static char job_keys[JOB_MAX_KEYS][JOB_KEY_BYTES];
static double job_values[JOB_MAX_KEYS];
static int job_nkeys;

static void job_load_map(void)
{
    FILE *f = fopen(ml_cardest_fname, "r");
    char line[JOB_KEY_BYTES + 100], key[JOB_KEY_BYTES], extra;
    double value;
    int i;
    if (!f) elog(ERROR, "JOB cannot open prediction file");
    job_nkeys = 0;
    while (fgets(line, sizeof(line), f))
    {
        if (sscanf(line, "%16383s %lf %c", key, &value, &extra) != 2 ||
            !isfinite(value) || (value != -1 && (value < 0 || value > 1)) ||
            job_nkeys >= JOB_MAX_KEYS)
        { fclose(f); elog(ERROR, "JOB malformed/oversized prediction map"); }
        for (i=0; i<job_nkeys; i++)
            if (strcmp(job_keys[i],key)==0)
            { fclose(f); elog(ERROR,"JOB duplicate prediction key"); }
        strlcpy(job_keys[job_nkeys],key,JOB_KEY_BYTES);
        job_values[job_nkeys++] = value;
    }
    if (ferror(f)) { fclose(f); elog(ERROR,"JOB prediction read error"); }
    fclose(f);
}

static double job_replace(PlannerInfo *root, Node *clause, double base)
{
    List *args;
    Oid opno, output;
    bool isarray=false, useor=false, varlen;
    Node *left, *right;
    Var *var;
    Const *constant;
    RangeTblEntry *rte;
    char *rel, *att, *op, *literal, *hex;
    StringInfoData column, key, trace;
    double result=base;
    int i, length;
    if (!ml_cardest_enabled && !print_single_tbl_queries) return base;
    if (IsA(clause,OpExpr)) { args=((OpExpr*)clause)->args; opno=((OpExpr*)clause)->opno; }
    else if (IsA(clause,ScalarArrayOpExpr))
    { args=((ScalarArrayOpExpr*)clause)->args;opno=((ScalarArrayOpExpr*)clause)->opno;
      isarray=true;useor=((ScalarArrayOpExpr*)clause)->useOr; }
    else return base;
    if (list_length(args)!=2) return base;
    left=linitial(args);right=lsecond(args);
    while (IsA(left,RelabelType)) left=(Node*)((RelabelType*)left)->arg;
    while (IsA(right,RelabelType)) right=(Node*)((RelabelType*)right)->arg;
    if (!IsA(left,Var) || !IsA(right,Const)) return base;
    var=(Var*)left;constant=(Const*)right;
    if (var->varlevelsup!=0 || var->varno<=0 || var->varno>list_length(root->parse->rtable)) return base;
    rte=rt_fetch(var->varno,root->parse->rtable);
    if (rte->rtekind!=RTE_RELATION || var->varattno<=0) return base;
    rel=get_rel_name(rte->relid);att=get_attname(rte->relid,var->varattno,false);
    initStringInfo(&column);appendStringInfo(&column,"%s.%s",rel,att);
    if (strcmp(column.data,"movie_companies.note") && strcmp(column.data,"title.title") &&
        strcmp(column.data,"name.name") && strcmp(column.data,"link_type.link")) return base;
    op=get_opname(opno);
    if (constant->constisnull || !op ||
        (strcmp(op,"=") && strcmp(op,"<>") && strcmp(op,"~~") && strcmp(op,"!~~")))
        elog(ERROR,"JOB unsupported target predicate");
    getTypeOutputInfo(constant->consttype,&output,&varlen);
    literal=OidOutputFunctionCall(output,constant->constvalue);
    initStringInfo(&key);
    appendStringInfo(&key,"%s|%s|%d|%d|%s",column.data,op,isarray?1:0,useor?1:0,literal);
    length=key.len;
    if (length*2+1>=JOB_KEY_BYTES) elog(ERROR,"JOB prediction key too long");
    hex=palloc(length*2+1);
    for (i=0;i<length;i++) sprintf(hex+2*i,"%02x",(unsigned char)key.data[i]);
    hex[length*2]='\0';
    if (ml_cardest_enabled)
    {
        bool found=false;
        if (query_no==0) job_load_map();
        for (i=0;i<job_nkeys;i++) if (strcmp(hex,job_keys[i])==0)
        { found=true;if(job_values[i]>=0) result=job_values[i];break; }
        if (!found) elog(ERROR,"JOB missing prediction key for %s",column.data);
        query_no++;
    }
    if (print_single_tbl_queries)
    {
        FILE *f=fopen("job_trace.jsonl","a");
        if (!f) elog(ERROR,"JOB cannot open trace");
        initStringInfo(&trace);
        appendStringInfoString(&trace,"{\"key\":");escape_json(&trace,hex);
        appendStringInfoString(&trace,",\"column\":");escape_json(&trace,column.data);
        appendStringInfoString(&trace,",\"alias\":");escape_json(&trace,rte->eref->aliasname);
        appendStringInfoString(&trace,",\"operator\":");escape_json(&trace,op);
        appendStringInfoString(&trace,",\"literal\":");escape_json(&trace,literal);
        appendStringInfo(&trace,",\"array\":%s,\"use_or\":%s,\"baseline\":%.17g,\"selected\":%.17g}",
            isarray?"true":"false",useor?"true":"false",base,result);
        if (fprintf(f,"%s\n",trace.data)<0) {fclose(f);elog(ERROR,"JOB trace write error");}
        fclose(f);pfree(trace.data);
    }
    return result;
}
