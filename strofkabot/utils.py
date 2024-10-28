import datetime
import discord
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

def parse_rpm_args(args):
    options = {
        'option': None,
        'flags': []
    }
    for arg in args:
        if arg.startswith('--'):
            options['flags'].append(arg)
        elif not options['option']:
            options['option'] = arg

    parsed = {
        'least': '--least' in options['flags'],
        'all_users': '--all' in options['flags'],
        'leaderboard': '--leaderboard' in options['flags']
    }
    return parsed

async def send_leaderboard(ctx, year, month, least, all_users, user_stats, bot):
    async def send_leaderboard_inner(year: int, month: int, month_offset: int = 0):
        adjusted_year, adjusted_month = adjust_month(year, month, month_offset)

        if datetime.date(adjusted_year, adjusted_month, 1) > datetime.date.today():
            await ctx.send("Cannot provide statistics for future months.")
            return

        stats = await user_stats.get_monthly_stats(adjusted_year, adjusted_month)
        if not stats:
            await ctx.send("No user statistics available for this month.")
            return

        filtered_stats = [stat for stat in stats if stat['total_msgs'] >= 30]

        if not filtered_stats:
            await ctx.send("No users with at least 30 messages found for this month.")
            return

        total_reactions = sum(stat['total_reacts'] for stat in stats)
        total_messages = sum(stat['total_msgs'] for stat in stats)
        server_avg_rpm = total_reactions / total_messages if total_messages > 0 else 0

        filtered_stats.sort(key=lambda x: x['avg_reacts'], reverse=not least)

        leaderboard_type = "Least" if least else "RPM"
        response = f"**{leaderboard_type} Leaderboard for {datetime.date(adjusted_year, adjusted_month, 1).strftime('%B %Y')}:**\n"
        response += f"Server Average RPM: {server_avg_rpm:.2f}\n"
        response += "(Users with at least 30 messages)\n```\n"
        response += f"{'User':<20} {'RPM':>5} {'Msgs':>5} {'Reacts':>7}\n"
        response += "-" * 40 + "\n"

        users_to_show = filtered_stats if all_users else filtered_stats[:10]

        for stat in users_to_show:
            display_name = stat['username'] or f"User {stat['author_id']}"
            response += f"{display_name[:20]:<20} {stat['avg_reacts']:5.2f} {stat['total_msgs']:5d} {stat['total_reacts']:7d}\n"
        response += "```"
        message = await ctx.send(response)
        await message.add_reaction("⬅️")
        await message.add_reaction("➡️")

        def check(reaction, user):
            return user == ctx.author and reaction.message.id == message.id and str(reaction.emoji) in ["⬅️", "➡️"]

        while True:
            try:
                reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=check)
                new_offset = month_offset
                if str(reaction.emoji) == "⬅️":
                    new_offset -= 1
                elif str(reaction.emoji) == "➡️":
                    new_offset += 1
                await message.delete()
                await send_leaderboard_inner(year, month, new_offset)
                break
            except asyncio.TimeoutError:
                break

    await send_leaderboard_inner(year, month)

async def send_personal_stats(ctx, year, month, user_stats):
    response = f"**Your RPM Stats for the past 12 months:**\n```\n"
    response += f"{'Month':<10} {'Messages':>10} {'Reactions':>10} {'RPM':>5}\n"
    response += "-" * 40 + "\n"

    current_date = datetime.datetime.now(datetime.timezone.utc)

    for i in range(12):
        month_date = current_date - datetime.timedelta(days=i*30)
        y, m = month_date.year, month_date.month
        user_stat = await user_stats.get_user_monthly_stats(ctx.author.id, y, m)
        if user_stat:
            response += f"{month_date.strftime('%b %Y'):<10} {user_stat['total_msgs']:>10} {user_stat['total_reacts']:>10} {user_stat['avg_reacts']:>5.2f}\n"
        else:
            response += f"{month_date.strftime('%b %Y'):<10} {'0':>10} {'0':>10} {'0.00':>5}\n"

    response += "```"

    await ctx.send(response)

def get_reply_info(message: discord.Message):
    reply_to_id = None
    reply_to_author = None
    reply_to_content = None

    if message.reference and message.reference.resolved:
        replied_msg = message.reference.resolved
        reply_to_id = replied_msg.id

        if isinstance(replied_msg, discord.DeletedReferencedMessage):
            reply_to_author = "Deleted User"
            reply_to_content = "Message was deleted"
        else:
            reply_to_author = replied_msg.author.display_name if replied_msg.author else "Unknown User"
            reply_to_content = replied_msg.content if hasattr(replied_msg, 'content') else "Content unavailable"

    return reply_to_id, reply_to_author, reply_to_content

async def fetch_inflation_data(user_stats):
    monthly_data = await user_stats.get_reaction_inflation_raw(monthly=True, limit=12)
    yearly_data = await user_stats.get_reaction_inflation_raw(monthly=False)
    return monthly_data, yearly_data

def calculate_monthly_inflation(monthly_data):
    monthly_inflation = []
    prev_avg = None
    for record in monthly_data:
        if prev_avg is not None:
            change = ((record['average_rpm'] - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0
            monthly_inflation.append({
                'month_year': f"{record['year']}-{record['month']:02d}",
                'change_percentage': round(change, 2),
                'average_rpm': record['average_rpm'],
                'total_reactions': record.get('total_reactions', 0),
                'total_messages': record.get('total_messages', 0)
            })
        prev_avg = record['average_rpm']
    return monthly_inflation

def calculate_yearly_inflation(yearly_data):
    yearly_inflation = []
    prev_avg = None
    for record in yearly_data:
        if prev_avg is not None:
            change = ((record['average_rpm'] - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0
            yearly_inflation.append({
                'year': record['year'],
                'change_percentage': round(change, 2),
                'average_rpm': record['average_rpm'],
                'total_reactions': record.get('total_reactions', 0),
                'total_messages': record.get('total_messages', 0)
            })
        prev_avg = record['average_rpm']
    return yearly_inflation

def create_monthly_inflation_plot(monthly_inflation):
    plt.figure(figsize=(10, 6))
    months = [record['month_year'] for record in monthly_inflation][::-1]
    changes = [record['change_percentage'] for record in monthly_inflation][::-1]
    averages = [record['average_rpm'] for record in monthly_inflation][::-1]
    
    ax = sns.barplot(x=months, y=changes, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.title("Monthly Reaction Inflation (Last 12 Months)")
    
    for i, v in enumerate(averages):
        ax.text(i, changes[i], f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def create_yearly_inflation_plot(yearly_inflation):
    plt.figure(figsize=(8, 6))
    years = [record['year'] for record in yearly_inflation]
    averages_yearly = [record['average_rpm'] for record in yearly_inflation]
    
    changes_yearly = [0]
    for i in range(1, len(averages_yearly)):
        yoy_change = ((averages_yearly[i] / averages_yearly[i-1]) - 1) * 100
        changes_yearly.append(yoy_change)
    
    ax = sns.barplot(x=years, y=changes_yearly, palette="deep")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=0)
    plt.xlabel("Year")
    plt.ylabel("YoY Inflation (%)")
    plt.title("Yearly Reaction Inflation")
    
    for i, v in enumerate(averages_yearly):
        ax.text(i, changes_yearly[i], f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def adjust_month(year: int, month: int, offset: int):
    new_month = month + offset
    new_year = year
    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -= 1
    return new_year, new_month

async def get_member_names(guild, member_ids):
    member_names = []
    for user_id in member_ids:
        member = guild.get_member(user_id)
        if member:
            name = ''.join(char for char in member.display_name if ord(char) < 128) or f"User {user_id}"
            member_names.append(name)
        else:
            member_names.append(f"User {user_id}")
    return member_names

def calculate_reaction_percentage(reaction_graph):
    row_sums = reaction_graph.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage = np.divide(reaction_graph, row_sums, where=row_sums!=0) * 100
        percentage = np.nan_to_num(percentage)
    return percentage

def determine_figure_size(num_users):
    return max(24, num_users * 0.6)

def generate_reaction_matrix_plot(data, labels, fig_size):
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, 
                     cmap="YlGnBu", square=True, cbar_kws={'shrink': .8})
    
    plt.xticks(rotation=90, ha='center', fontsize=16)
    plt.yticks(rotation=0, va='center', fontsize=16)
    
    plt.title("Reaction Matrix (Percentage of Reactions Given)", fontsize=22, pad=20)
    plt.xlabel("Reactions Received From", fontsize=18, labelpad=15)
    plt.ylabel("Reactions Given To", fontsize=18, labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

async def get_non_bot_member_ids(guild):
    return [member.id for member in guild.members if not member.bot]

def prepare_clustering_data(guild, reactions_data):
    data = []
    member_names = []
    for user_id, stats in reactions_data.items():
        data.append([stats.get('given', 0), stats.get('received', 0)])
        member = guild.get_member(user_id)
        member_names.append(member.display_name if member else f"User {user_id}")
    return np.array(data), member_names

def perform_kmeans_clustering(data, num_clusters=4):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data_normalized)
    return labels

def generate_cluster_plot(data, labels, member_names):
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(sns.color_palette("hsv", np.unique(labels).size).as_hex())
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=labels, palette=cmap, legend='full')
    plt.xlabel("Reactions Given")
    plt.ylabel("Reactions Received")
    plt.title("User Clusters Based on Reactions")
    plt.legend(title='Cluster')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


# @commands.command(name='repopulate_stats')
# async def repopulate_user_stats(self, ctx: commands.Context):
#     """Repopulate the user_stats_monthly table by processing all reactions monthly."""
#     self.logger.info("Starting repopulation of user_stats_monthly.")
#     await ctx.send("Repopulating user statistics based on reactions. This may take some time...")

#     try:
#         await self.repopulate_user_stats_method()
#         await ctx.send("Repopulation of user statistics completed successfully.")
#         self.logger.info("Repopulation of user_stats_monthly completed successfully.")
#     except Exception as e:
#         self.logger.exception(f"Error during repopulation: {e}")
#         await ctx.send("An error occurred during repopulation. Check logs for details.")

# async def repopulate_user_stats_method(self):
#     """Repopulate the user_stats_monthly table by processing all historical reactions monthly."""
#     if not self.guild:
#         self.logger.warning("Guild not set. Skipping repopulation.")
#         return

#     total_reaction_count = 0

#     async def process_channel(channel: discord.TextChannel):
#         nonlocal total_reaction_count
#         self.logger.info(f"Processing channel: {channel.name} (ID: {channel.id}) for repopulation.")

#         stats_to_update = []
#         try:
#             async for message in channel.history(after=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc), before=None, limit=None):
#                 if message.author.bot:
#                     continue
#                 if not message.content.strip():
#                     continue

#                 reaction_count = self._get_all_reacts(message)
#                 for reaction in message.reactions:
#                     async for user in reaction.users():
#                         if user.bot:
#                             continue
#                         await self.user_stats.update_reaction_stats(user.id, message.author.id, message.created_at)
#                         total_reaction_count += reaction.count

#                 stats_to_update.append((message.author.id, reaction_count, message.created_at))

#                 if len(stats_to_update) >= 500:
#                     await self.user_stats.batch_update_stats(stats_to_update)
#                     stats_to_update.clear()

#             if stats_to_update:
#                 await self.user_stats.batch_update_stats(stats_to_update)

#         except Exception as e:
#             self.logger.exception(f"Error processing channel {channel.name}: {e}")

#         self.logger.info(f"Channel {channel.name} processed.")

#     channels = [channel for channel in self.guild.text_channels if channel.permissions_for(self.guild.me).read_messages]
#     await asyncio.gather(*(process_channel(channel) for channel in channels))

#     self.logger.info(f"Repopulation completed. Total reactions processed: {total_reaction_count}")

