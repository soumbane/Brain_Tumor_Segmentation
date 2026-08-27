-- =====================================================================
-- BraTS 2023 project: Snowflake prerequisites
--
-- MUST BE RUN BY A ROLE WITH ACCOUNTADMIN (or equivalent delegated
-- privileges). The project owner, SOUMYANIL.BANERJEE@COREWELLHEALTH.ORG,
-- currently holds only PUBLIC and SFK_CORP_JIRA, neither of which grants
-- CREATE COMPUTE POOL (an account-level privilege). Verified 2026-08-27:
--
--   SHOW GRANTS TO USER "SOUMYANIL.BANERJEE@COREWELLHEALTH.ORG";
--     -> PUBLIC, SFK_CORP_JIRA only
--   SHOW EXTERNAL ACCESS INTEGRATIONS;
--     -> no rows (no PyPI egress available)
--   SHOW COMPUTE POOLS;
--     -> SYSTEM_COMPUTE_POOL_GPU is GPU_NV_S (1x A10G) and SUSPENDED
--
-- Phases 0-1 of the project (extraction, manifest, EDA, splits) do not
-- require any of this. Phases 2+ (any GPU work) are blocked until it runs.
-- =====================================================================

USE ROLE ACCOUNTADMIN;

-- ---------------------------------------------------------------------
-- 1. A dedicated role for the project
-- ---------------------------------------------------------------------
CREATE ROLE IF NOT EXISTS BRATS_ML_ROLE
  COMMENT = 'BraTS 2023 multi-task segmentation/classification project';

GRANT ROLE BRATS_ML_ROLE TO USER "SOUMYANIL.BANERJEE@COREWELLHEALTH.ORG";

-- ---------------------------------------------------------------------
-- 2. GPU compute pool: GPU_NV_M = 4x NVIDIA A10G 24GB, 44 vCPU, 178 GiB
--    RAM, 93.13 GiB local disk. Confirmed available in this account via
--    SHOW COMPUTE POOL INSTANCE FAMILIES (region AWS_US_EAST_1).
--
--    Created INITIALLY_SUSPENDED so it bills nothing until first use.
--    AUTO_SUSPEND_SECS is deliberately short: A10G nodes idle expensively.
-- ---------------------------------------------------------------------
CREATE COMPUTE POOL IF NOT EXISTS BRATS_GPU_NV_M
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  AUTO_SUSPEND_SECS = 600
  COMMENT = 'BraTS 2023 training/inference: 4x A10G';

GRANT USAGE, MONITOR ON COMPUTE POOL BRATS_GPU_NV_M TO ROLE BRATS_ML_ROLE;

-- Optional: let the project owner size pools themselves instead of the above.
-- GRANT CREATE COMPUTE POOL ON ACCOUNT TO ROLE BRATS_ML_ROLE;

-- ---------------------------------------------------------------------
-- 3. PyPI egress. Required because the container runtime does not ship
--    MONAI, nibabel, or the BraTS metrics dependencies.
-- ---------------------------------------------------------------------
CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS PYPI_EAI
  ALLOWED_NETWORK_RULES = (snowflake.external_access.pypi_rule)
  ENABLED = TRUE
  COMMENT = 'PyPI access for ML Jobs payload dependencies';

GRANT USAGE ON INTEGRATION PYPI_EAI TO ROLE BRATS_ML_ROLE;

-- ---------------------------------------------------------------------
-- 3b. OPTIONAL: Weights & Biases live logging.
--
--     Only needed for `--wandb-mode online`. The default is OFFLINE, which
--     writes run data to the container's local disk and requires no network
--     at all -- curves are viewed later via `wandb sync`. Skip this whole
--     section if egress to a third-party SaaS is not acceptable.
-- ---------------------------------------------------------------------
CREATE OR REPLACE NETWORK RULE WANDB_NR
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('api.wandb.ai');

CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS WANDB_EAI
  ALLOWED_NETWORK_RULES = (WANDB_NR)
  ENABLED = TRUE
  COMMENT = 'Weights & Biases live experiment tracking';

GRANT USAGE ON INTEGRATION WANDB_EAI TO ROLE BRATS_ML_ROLE;

-- The API key is passed as a SECRET so it never appears in job arguments,
-- the payload, or the query history.
CREATE SECRET IF NOT EXISTS BRATS_MRI.CORE.WANDB_API_KEY
  TYPE = GENERIC_STRING
  SECRET_STRING = '<paste-the-wandb-api-key>'
  COMMENT = 'W&B API key for ML Jobs (online tracking only)';

GRANT USAGE ON SECRET BRATS_MRI.CORE.WANDB_API_KEY TO ROLE BRATS_ML_ROLE;

-- ---------------------------------------------------------------------
-- 4. Project database, schema, and stages
--
--    Data volume: 32.9 GB labeled (2350 cases) + 5.7 GB unlabeled
--    (405 cases), as .nii.gz. Stage storage is billed; see
--    scripts/README_costs.md before uploading.
-- ---------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS BRATS_MRI;
GRANT OWNERSHIP ON DATABASE BRATS_MRI TO ROLE BRATS_ML_ROLE;

USE DATABASE BRATS_MRI;
CREATE SCHEMA IF NOT EXISTS CORE;
GRANT OWNERSHIP ON SCHEMA BRATS_MRI.CORE TO ROLE BRATS_ML_ROLE;

-- Warehouse for the Snowpark session that submits ML Jobs (metadata only;
-- no heavy SQL runs here).
GRANT USAGE ON WAREHOUSE CORP_ENTITY_XS_WH TO ROLE BRATS_ML_ROLE;

-- ---------------------------------------------------------------------
-- 5. Cost guardrail. Do NOT skip: GPU_NV_M credit rates are unverified
--    against the Service Consumption Table, and a forgotten resumed pool
--    is the most likely way this project overspends.
-- ---------------------------------------------------------------------
-- Attach BRATS_GPU_NV_M to a budget, e.g.:
--
--   USE SCHEMA SNOWFLAKE.LOCAL;
--   CREATE OR REPLACE SNOWFLAKE.CORE.BUDGET BRATS_BUDGET();
--   CALL BRATS_BUDGET!SET_SPENDING_LIMIT(<credits_per_month>);
--   CALL BRATS_BUDGET!ADD_RESOURCE(SYSTEM$REFERENCE('COMPUTE_POOL',
--                                  'BRATS_GPU_NV_M', 'SESSION', 'MODIFY'));
--   CALL BRATS_BUDGET!SET_NOTIFICATION_USERS(
--          ARRAY_CONSTRUCT('SOUMYANIL.BANERJEE@COREWELLHEALTH.ORG'));

-- ---------------------------------------------------------------------
-- 6. Verification: run these AS BRATS_ML_ROLE to confirm the grants took.
-- ---------------------------------------------------------------------
-- USE ROLE BRATS_ML_ROLE;
-- SHOW COMPUTE POOLS LIKE 'BRATS_GPU_NV_M';
-- SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI_EAI';
-- SELECT CURRENT_ROLE(), CURRENT_DATABASE();
