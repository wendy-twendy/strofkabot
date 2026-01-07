"""
Tests for the UserStats class.
"""
import datetime

from strofkabot.user_stats import UserStats


class TestUserStatsInitialization:
    """Tests for UserStats initialization."""

    async def test_initialize_creates_all_tables(self, user_stats_db: UserStats):
        """Verify all required tables are created."""
        async with user_stats_db.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]

        assert 'user_stats_monthly' in table_names
        assert 'user_mapping' in table_names
        assert 'user_reactions_monthly' in table_names

    async def test_database_connection_initialized(self, user_stats_db: UserStats):
        """Test that the database connection is properly initialized."""
        assert user_stats_db.db is not None
        assert user_stats_db.db.conn is not None


class TestStatsAggregation:
    """Tests for statistics update and aggregation."""

    async def test_update_stats_creates_new_record(self, user_stats_db: UserStats):
        """Test that update_stats creates a new monthly record."""
        await user_stats_db.update_stats(
            author_id=99999,
            reaction_count=5,
            timestamp=datetime.datetime(2024, 3, 15)
        )

        async with user_stats_db.db.conn.execute(
            "SELECT total_messages, total_reactions FROM user_stats_monthly "
            "WHERE author_id = ? AND year = ? AND month = ?",
            (99999, 2024, 3)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1  # total_messages
        assert row[1] == 5  # total_reactions

    async def test_update_stats_increments_existing(self, user_stats_db: UserStats):
        """Test that subsequent updates increment counters."""
        ts = datetime.datetime(2024, 4, 1)
        await user_stats_db.update_stats(88888, 3, ts)
        await user_stats_db.update_stats(88888, 7, ts)

        async with user_stats_db.db.conn.execute(
            "SELECT total_messages, total_reactions FROM user_stats_monthly "
            "WHERE author_id = ? AND year = ? AND month = ?",
            (88888, 2024, 4)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 2   # 2 messages
        assert row[1] == 10  # 3 + 7 reactions

    async def test_batch_update_stats(self, user_stats_db: UserStats):
        """Test batch updating stats."""
        stats = [
            (77777, 2, datetime.datetime(2024, 5, 1)),
            (77777, 3, datetime.datetime(2024, 5, 2)),
            (77778, 5, datetime.datetime(2024, 5, 1)),
        ]
        await user_stats_db.batch_update_stats(stats)

        # Check 77777 aggregated correctly
        async with user_stats_db.db.conn.execute(
            "SELECT total_messages, total_reactions FROM user_stats_monthly "
            "WHERE author_id = ? AND year = ? AND month = ?",
            (77777, 2024, 5)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 2  # 2 messages
        assert row[1] == 5  # 2 + 3 reactions


class TestMonthlyQueries:
    """Tests for monthly statistics queries."""

    async def test_get_monthly_stats(self, populated_user_stats: UserStats):
        """Test retrieving monthly stats for all users."""
        stats = await populated_user_stats.get_monthly_stats(2024, 1)

        assert len(stats) == 3  # Alice, Bob, Charlie

        # Check structure
        for stat in stats:
            assert 'author_id' in stat
            assert 'username' in stat
            assert 'total_msgs' in stat
            assert 'total_reacts' in stat
            assert 'avg_reacts' in stat

    async def test_get_monthly_stats_empty_month(self, user_stats_db: UserStats):
        """Test that querying empty month returns empty list."""
        stats = await user_stats_db.get_monthly_stats(2099, 12)
        assert stats == []

    async def test_get_user_monthly_stats(self, populated_user_stats: UserStats):
        """Test getting stats for a specific user."""
        stat = await populated_user_stats.get_user_monthly_stats(12345, 2024, 1)

        assert stat is not None
        assert stat['total_msgs'] == 2  # Two messages in Jan 2024
        assert stat['total_reacts'] == 8  # 5 + 3

    async def test_get_user_monthly_stats_nonexistent(self, user_stats_db: UserStats):
        """Test that nonexistent user returns None."""
        stat = await user_stats_db.get_user_monthly_stats(99999999, 2024, 1)
        assert stat is None


class TestReactionNetwork:
    """Tests for reaction tracking and network queries."""

    async def test_update_reaction_stats(self, user_stats_db: UserStats):
        """Test recording a reaction between users."""
        await user_stats_db.update_reaction_stats(
            giver_id=1111,
            receiver_id=2222,
            timestamp=datetime.datetime(2024, 6, 15)
        )

        async with user_stats_db.db.conn.execute(
            "SELECT reaction_count FROM user_reactions_monthly "
            "WHERE giver_id = ? AND receiver_id = ? AND year = ? AND month = ?",
            (1111, 2222, 2024, 6)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1

    async def test_reaction_stats_increment(self, user_stats_db: UserStats):
        """Test that multiple reactions increment the counter."""
        ts = datetime.datetime(2024, 7, 1)
        for _ in range(5):
            await user_stats_db.update_reaction_stats(3333, 4444, ts)

        async with user_stats_db.db.conn.execute(
            "SELECT reaction_count FROM user_reactions_monthly "
            "WHERE giver_id = ? AND receiver_id = ?",
            (3333, 4444)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 5

    async def test_batch_update_reaction_stats(self, user_stats_db: UserStats):
        """Test batch updating reaction stats."""
        stats = [
            (1111, 2222, datetime.datetime(2024, 8, 1)),
            (1111, 2222, datetime.datetime(2024, 8, 2)),
            (1111, 3333, datetime.datetime(2024, 8, 1)),
        ]
        await user_stats_db.batch_update_reaction_stats(stats)

        # Check 1111 -> 2222 aggregated correctly
        async with user_stats_db.db.conn.execute(
            "SELECT reaction_count FROM user_reactions_monthly "
            "WHERE giver_id = ? AND receiver_id = ? AND year = ? AND month = ?",
            (1111, 2222, 2024, 8)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 2  # 2 reactions


class TestInflationData:
    """Tests for inflation data retrieval."""

    async def test_get_reaction_inflation_raw_monthly(self, populated_user_stats: UserStats):
        """Test retrieving monthly inflation data."""
        data = await populated_user_stats.get_reaction_inflation_raw(monthly=True, limit=12)

        assert isinstance(data, list)
        if data:  # If there's data
            record = data[0]
            assert 'year' in record
            assert 'month' in record
            assert 'average_rpm' in record

    async def test_get_reaction_inflation_raw_yearly(self, populated_user_stats: UserStats):
        """Test retrieving yearly inflation data."""
        data = await populated_user_stats.get_reaction_inflation_raw(monthly=False)

        assert isinstance(data, list)
        if data:
            record = data[0]
            assert 'year' in record
            assert 'average_rpm' in record


class TestUserMapping:
    """Tests for user mapping operations."""

    async def test_update_user_mapping(self, user_stats_db: UserStats):
        """Test creating/updating user mapping."""
        await user_stats_db.update_user_mapping(55555, "TestUser")

        async with user_stats_db.db.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (55555,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "TestUser"

    async def test_update_user_mapping_overwrites(self, user_stats_db: UserStats):
        """Test that updating mapping overwrites previous name."""
        await user_stats_db.update_user_mapping(66666, "OldName")
        await user_stats_db.update_user_mapping(66666, "NewName")

        async with user_stats_db.db.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (66666,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "NewName"


class TestResetStats:
    """Tests for statistics reset functionality."""

    async def test_reset_stats_clears_tables(self, populated_user_stats: UserStats):
        """Test that reset_stats clears the appropriate tables."""
        await populated_user_stats.reset_stats()

        async with populated_user_stats.db.conn.execute(
            "SELECT COUNT(*) FROM user_stats_monthly"
        ) as cursor:
            stats_count = (await cursor.fetchone())[0]

        async with populated_user_stats.db.conn.execute(
            "SELECT COUNT(*) FROM user_reactions_monthly"
        ) as cursor:
            reactions_count = (await cursor.fetchone())[0]

        assert stats_count == 0
        assert reactions_count == 0


class TestGDPData:
    """Tests for GDP data retrieval."""

    async def test_get_gdp_data_returns_aggregated_data(self, populated_user_stats: UserStats):
        """Test that get_gdp_data returns aggregated message counts per month."""
        data = await populated_user_stats.get_gdp_data(limit=24)

        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure
        for record in data:
            assert 'year' in record
            assert 'month' in record
            assert 'total_messages' in record

    async def test_get_gdp_data_respects_limit(self, populated_user_stats: UserStats):
        """Test that get_gdp_data respects the limit parameter."""
        data = await populated_user_stats.get_gdp_data(limit=1)
        assert len(data) <= 1

    async def test_get_gdp_data_empty_database(self, user_stats_db: UserStats):
        """Test that get_gdp_data returns empty list for empty database."""
        data = await user_stats_db.get_gdp_data()
        assert data == []


class TestReactionTradeData:
    """Tests for reaction trade data retrieval."""

    async def test_get_reaction_trade_data_returns_correct_structure(
        self, populated_user_stats: UserStats
    ):
        """Test that get_reaction_trade_data returns correct structure."""
        data = await populated_user_stats.get_reaction_trade_data(
            user_id=12345,
            year=2024,
            month=1,
            limit=5
        )

        assert isinstance(data, dict)
        assert 'exports' in data
        assert 'imports' in data
        assert 'total_given' in data
        assert 'total_received' in data
        assert 'trade_balance' in data

    async def test_get_reaction_trade_data_exports_format(
        self, populated_user_stats: UserStats
    ):
        """Test that exports contain (receiver_id, count) tuples."""
        data = await populated_user_stats.get_reaction_trade_data(
            user_id=12345,
            year=2024,
            month=1
        )

        for export in data['exports']:
            assert isinstance(export, tuple)
            assert len(export) == 2
            assert isinstance(export[0], int)  # receiver_id
            assert isinstance(export[1], int)  # count

    async def test_get_reaction_trade_data_trade_balance_calculation(
        self, populated_user_stats: UserStats
    ):
        """Test that trade balance is calculated correctly."""
        data = await populated_user_stats.get_reaction_trade_data(
            user_id=12345,
            year=2024,
            month=1
        )

        expected_balance = data['total_received'] - data['total_given']
        assert data['trade_balance'] == expected_balance

    async def test_get_reaction_trade_data_nonexistent_user(
        self, user_stats_db: UserStats
    ):
        """Test that nonexistent user returns zero totals."""
        data = await user_stats_db.get_reaction_trade_data(
            user_id=99999999,
            year=2024,
            month=1
        )

        assert data['exports'] == []
        assert data['imports'] == []
        assert data['total_given'] == 0
        assert data['total_received'] == 0


class TestReactionNetworkForMonth:
    """Tests for reaction network data retrieval."""

    async def test_get_reaction_network_returns_correct_structure(
        self, populated_user_stats: UserStats
    ):
        """Test that get_reaction_network_for_month returns correct structure."""
        data = await populated_user_stats.get_reaction_network_for_month(2024, 1)

        assert isinstance(data, list)
        if data:
            record = data[0]
            assert 'giver_username' in record
            assert 'receiver_username' in record
            assert 'reaction_count' in record
            assert 'giver_messages' in record
            assert 'receiver_messages' in record

    async def test_get_reaction_network_empty_month(self, user_stats_db: UserStats):
        """Test that empty month returns empty list."""
        data = await user_stats_db.get_reaction_network_for_month(2099, 12)
        assert data == []

    async def test_get_reaction_network_includes_usernames(
        self, populated_user_stats: UserStats
    ):
        """Test that usernames are resolved from user_mapping."""
        data = await populated_user_stats.get_reaction_network_for_month(2024, 1)

        # The fixture sets up reactions between 12345 (Alice), 67890 (Bob), 11111 (Charlie)
        if data:
            for record in data:
                assert record['giver_username'] in ['Alice', 'Bob', 'Charlie']
                assert record['receiver_username'] in ['Alice', 'Bob', 'Charlie']
