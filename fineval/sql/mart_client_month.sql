-- Витрина «клиент × месяц» для оценки эффекта инициативы.
-- Один и тот же SQL исполняется в DuckDB (локально, тесты) и в Spark SQL (прод-объёмы):
-- используем только общий для обоих диалектов набор функций.
--
-- Параметры (подставляются раннером):
--   :pilot_start   — первый месяц пилота, YYYYMM
--   :interchange   — ставка interchange-дохода, доля от оборота
--   :ftp_spread    — годовой трансфертный спред на остатки
--
-- Источники (регистрируются раннером как view): regions, clients, card_activity, balances, cashback

WITH base AS (
    SELECT
        a.client_id,
        a.month,
        c.region_id,
        c.segment,
        r.is_pilot,
        CASE WHEN a.month >= :pilot_start THEN 1 ELSE 0 END                     AS post,
        (CAST(FLOOR(a.month / 100) AS INT) - CAST(FLOOR(:pilot_start / 100) AS INT)) * 12
          + (a.month % 100) - (:pilot_start % 100)                                AS rel_month,
        a.spend,
        a.category_spend,
        a.is_active,
        COALESCE(b.avg_balance, 0)                                                AS avg_balance,
        COALESCE(k.cashback_cost, 0)                                              AS cashback_cost,
        COALESCE(k.partner_compensation, 0)                                       AS partner_compensation,
        COALESCE(k.opted_in, 0)                                                   AS opted_in
    FROM card_activity a
    JOIN clients  c ON c.client_id = a.client_id
    JOIN regions  r ON r.region_id = c.region_id
    LEFT JOIN balances b ON b.client_id = a.client_id AND b.month = a.month
    LEFT JOIN cashback k ON k.client_id = a.client_id AND k.month = a.month
)
SELECT
    *,
    spend * :interchange                                     AS interchange_income,
    avg_balance * :ftp_spread / 12                           AS nii_income,
    cashback_cost - partner_compensation                     AS net_cashback_cost,
    spend * :interchange + avg_balance * :ftp_spread / 12
      - (cashback_cost - partner_compensation)               AS net_income,
    is_pilot * post                                          AS treated_post
FROM base
